"""Export one completed LLM run as the user/post JSON expected by build_table.py.

From the repository root:
    python project_new/src/02_llm_analysis/aggregate.py --run-dir project_new/data/llm_analysis/<run-id>

The default output is <run-dir>/llm_analysis.json. It contains only successful
posts; single-post content rejections remain auditable in state.jsonl. An
unfinished or inconsistent run produces no replacement output.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"需要 JSON 对象：{path}")
    return value


def _scan_input(input_dir: Path, run: dict) -> tuple[list[str], dict[str, list[str]]]:
    users: list[str] = []
    user_ids: set[str] = set()
    posts: dict[str, list[tuple[datetime, str]]] = defaultdict(list)
    post_ids: set[str] = set()
    paths = {name: input_dir / f"{name}_cleaned.jsonl" for name in ("user", "weibo")}
    for name, path in paths.items():
        if _digest(path) != run.get(f"{name}_sha256"):
            raise ValueError(f"输入摘要与 run.json 不一致：{path}")
        with path.open("r", encoding="utf-8-sig") as stream:
            for number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                    uid = str(row["uid"])
                    if not uid.isdecimal():
                        raise ValueError("用户 ID 无效")
                    if name == "user":
                        if uid in user_ids:
                            raise ValueError("用户 ID 重复")
                        user_ids.add(uid)
                        users.append(uid)
                    else:
                        wid = str(row["wid"])
                        if not wid.isdecimal() or wid in post_ids:
                            raise ValueError("微博 ID 无效或重复")
                        post_ids.add(wid)
                        posts[uid].append((datetime.fromisoformat(row["created_at"]), wid))
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"{path}:{number}: {error}") from error
    if set(posts) != user_ids:
        raise ValueError("清洗输入的用户与微博关联不完整")
    if len(users) != run.get("users") or len(post_ids) != run.get("weibos"):
        raise ValueError("清洗输入行数与 run.json 不一致")
    ordered: dict[str, list[str]] = {}
    for uid, rows in posts.items():
        try:
            rows.sort(key=lambda row: (row[0], row[1]))
        except TypeError as error:
            raise ValueError(f"用户 {uid} 的时间混用了带时区与不带时区格式") from error
        ordered[uid] = [wid for _, wid in rows]
    return users, ordered


def _latest_state(path: Path) -> dict[tuple[str, str], dict]:
    latest: dict[tuple[str, str], dict] = {}
    with path.open("rb") as stream:
        for number, line in enumerate(stream, 1):
            if not line.endswith(b"\n"):
                raise ValueError(f"{path}:{number}: 状态记录未完整写入，请先续跑分析")
            try:
                event = json.loads(line)
                uid, batch_id = event["uid"], event["batch_id"]
                if not isinstance(uid, str) or not isinstance(batch_id, str) or not isinstance(event["wids"], list):
                    raise ValueError("批次标识或微博列表无效")
                if event["status"] not in {"succeeded", "content_rejected", "retryable_error", "invalid_response"}:
                    raise ValueError("批次状态无效")
                latest[(uid, batch_id)] = event
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error
    return latest


def _collect_batch(
    run_dir: Path, latest: dict, visited: set, uid: str, batch_id: str,
    wids: list[str], items: dict, excluded: list[str],
) -> None:
    key = (uid, batch_id)
    event = latest.get(key)
    if event is None:
        raise ValueError(f"批次 {uid}/{batch_id} 尚未完成；请先续跑分析")
    visited.add(key)
    if event["wids"] != wids:
        raise ValueError(f"批次 {uid}/{batch_id} 的微博 ID 与输入不一致")
    status = event["status"]
    if status == "content_rejected":
        if len(wids) == 1 and event.get("scope") == "post":
            excluded.extend(wids)
            return
        if len(wids) > 1 and event.get("scope") == "batch":
            middle = len(wids) // 2
            _collect_batch(run_dir, latest, visited, uid, batch_id + ".1", wids[:middle], items, excluded)
            _collect_batch(run_dir, latest, visited, uid, batch_id + ".2", wids[middle:], items, excluded)
            return
        raise ValueError(f"批次 {uid}/{batch_id} 的审核排除范围无效")
    if status != "succeeded":
        raise ValueError(f"批次 {uid}/{batch_id} 仍为 {status}；请先续跑分析")

    result_path = run_dir / "results" / uid / f"{batch_id}.json"
    expected_hash = event.get("result_sha256")
    if not isinstance(expected_hash, str) or _digest(result_path) != expected_hash:
        raise ValueError(f"结果文件摘要不匹配：{result_path}")
    result = _load_json(result_path)
    batch_items = result.get("items")
    if (result.get("uid") != uid or result.get("batch_id") != batch_id
            or result.get("wids") != wids or not isinstance(batch_items, dict)
            or set(batch_items) != set(wids)):
        raise ValueError(f"结果文件微博 ID 覆盖不完整：{result_path}")
    for wid, item in batch_items.items():
        if (wid in items or not isinstance(item, dict)
                or not isinstance(item.get("comprehensive_sentiment"), dict)
                or not item["comprehensive_sentiment"].get("type")):
            raise ValueError(f"结果文件中微博 {wid} 重复或情感字段无效：{result_path}")
        items[wid] = item


@contextmanager
def _run_lock(run_dir: Path):
    """Use the analysis runner's lock so a run cannot change during export."""
    with (run_dir / ".run.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            if stream.seek(0, os.SEEK_END) == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("该实验目录已有分析进程") from error
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError("该实验目录已有分析进程") from error
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def aggregate_run(run_dir: Path, output_path: Path | None = None) -> dict:
    """Atomically export a complete run without changing its inputs or state."""
    if not run_dir.is_dir():
        raise ValueError(f"实验目录不存在：{run_dir}")
    output_path = output_path or run_dir / "llm_analysis.json"
    with _run_lock(run_dir):
        run = _load_json(run_dir / "run.json")
        batch_size = run.get("batch_size")
        if run.get("schema_version") != 1 or type(batch_size) is not int or batch_size < 1:
            raise ValueError("不支持的 run.json 格式或批次大小")
        input_dir = Path(run["input_dir"])
        protected = {run_dir / "run.json", run_dir / "state.jsonl", run_dir / ".run.lock",
                     input_dir / "user_cleaned.jsonl", input_dir / "weibo_cleaned.jsonl"}
        resolved_output = output_path.resolve()
        if (resolved_output in {path.resolve() for path in protected}
                or resolved_output.is_relative_to((run_dir / "results").resolve())):
            raise ValueError("汇总输出不能覆盖实验状态、批次结果或清洗输入")
        users, posts = _scan_input(input_dir, run)
        latest = _latest_state(run_dir / "state.jsonl")
        visited: set[tuple[str, str]] = set()
        report = {"users": len(users), "succeeded_posts": 0, "excluded_posts": 0,
                  "output": str(output_path)}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write("{\n")
                for index, uid in enumerate(users):
                    wids = posts[uid]
                    items: dict = {}
                    excluded: list[str] = []
                    for number, start in enumerate(range(0, len(wids), batch_size), 1):
                        _collect_batch(run_dir, latest, visited, uid, str(number),
                                       wids[start:start + batch_size], items, excluded)
                    if len(items) + len(excluded) != len(wids):
                        raise ValueError(f"用户 {uid} 的微博覆盖不完整")
                    if index:
                        stream.write(",\n")
                    json.dump(uid, stream, ensure_ascii=False)
                    stream.write(": ")
                    json.dump(items, stream, ensure_ascii=False)
                    report["succeeded_posts"] += len(items)
                    report["excluded_posts"] += len(excluded)
                if set(latest) != visited:
                    raise ValueError("state.jsonl 含有当前输入之外的批次")
                stream.write("\n}\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, output_path)
        finally:
            temporary.unlink(missing_ok=True)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="已完成的 LLM 实验目录")
    parser.add_argument("--output", type=Path, help="汇总 JSON 路径，默认写入实验目录")
    args = parser.parse_args()
    print(json.dumps(aggregate_run(args.run_dir, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
