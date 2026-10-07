"""Analyze cleaned local JSONL with one DeepSeek model and resumable batches."""

from __future__ import annotations

import argparse
import contextlib
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Callable

from prompt import PROMPT


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "datasets" / "legacy"
DEFAULT_RUNS = PROJECT_ROOT / "data" / "llm_analysis"
SCHEMA_VERSION = 1
BATCH_SIZE = 20


class ContentRejected(Exception):
    """The provider refused to analyze this request's content."""

    def __init__(self, code: str = "content_filter", request_id: str | None = None):
        super().__init__(code)
        self.code = code
        self.request_id = request_id


class InvalidResponse(Exception):
    """A model response is missing or does not cover the requested posts."""


def _digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def scan_dataset(input_dir: Path) -> tuple[list[str], dict[str, list[int]], dict]:
    """Validate input and index byte offsets without retaining all post text."""
    users: list[str] = []
    user_ids: set[str] = set()
    offsets: dict[str, list[int]] = defaultdict(list)
    post_ids: set[str] = set()
    digests = {}
    for filename in ("user_cleaned.jsonl", "weibo_cleaned.jsonl"):
        digest = hashlib.sha256()
        with (input_dir / filename).open("rb") as stream:
            for number, line in enumerate(stream, 1):
                position = stream.tell() - len(line)
                digest.update(line)
                try:
                    row = json.loads(line)
                    uid = str(row["uid"])
                    if not uid.isdecimal():
                        raise ValueError("invalid uid")
                    if filename.startswith("user"):
                        if uid in user_ids:
                            raise ValueError("duplicate uid")
                        user_ids.add(uid)
                        users.append(uid)
                    else:
                        wid = str(row["wid"])
                        if not wid.isdecimal() or wid in post_ids:
                            raise ValueError("invalid or duplicate wid")
                        if not isinstance(row["text"], str):
                            raise ValueError("text is not a string")
                        datetime.fromisoformat(row["created_at"])
                        post_ids.add(wid)
                        offsets[uid].append(position)
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(f"{filename}:{number}: {error}") from error
        digests["user" if filename.startswith("user") else "weibo"] = digest.hexdigest()
    if set(offsets) != user_ids:
        raise ValueError(f"用户/微博关联不完整：孤立用户 {len(user_ids - set(offsets))}，孤立微博用户 {len(set(offsets) - user_ids)}")
    manifest_path = input_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("user_sha256") != digests["user"] or manifest.get("weibo_sha256") != digests["weibo"]:
            raise ValueError("输入文件与 manifest.json 的摘要不一致")
        if manifest.get("users") != len(users) or manifest.get("weibos") != len(post_ids):
            raise ValueError("输入文件与 manifest.json 的行数不一致")
    return users, offsets, {
        "user_sha256": digests["user"],
        "weibo_sha256": digests["weibo"],
        "users": len(users),
        "weibos": len(post_ids),
        "batches": sum((len(positions) + BATCH_SIZE - 1) // BATCH_SIZE for positions in offsets.values()),
    }


def load_user_posts(stream, positions: list[int]) -> list[dict]:
    posts = []
    for position in positions:
        stream.seek(position)
        row = json.loads(stream.readline())
        row["uid"] = str(row["uid"])
        row["wid"] = str(row["wid"])
        row["_time"] = datetime.fromisoformat(row["created_at"])
        posts.append(row)
    try:
        posts.sort(key=lambda post: (post["_time"], post["wid"]))
    except TypeError as error:
        raise ValueError("同一用户的发布时间混用了有时区与无时区格式") from error
    for index, post in enumerate(posts):
        post["post_interval"] = "-" if index == 0 else f"{(post['_time'] - posts[index - 1]['_time']).total_seconds() / 86400:.2f}天"
        post["mentions"] = re.findall(r"@([^\s]+)", post["text"])
        post["topics"] = re.findall(r"#([^#]+)#", post["text"])
    return posts


def _input_payload(posts: list[dict]) -> list[dict]:
    return [
        {"微博ID": post["wid"], "微博内容": post["text"], "发布时间": post["_time"].strftime("%Y-%m-%d %H:%M:%S")}
        for post in posts
    ]


def format_response(posts: list[dict], response: dict) -> dict:
    if not isinstance(response, dict) or not isinstance(response.get("analysis"), list):
        raise InvalidResponse("缺少 analysis 列表")
    wanted = {post["wid"] for post in posts}
    found = {}
    for item in response["analysis"]:
        if not isinstance(item, dict) or "微博ID" not in item:
            raise InvalidResponse("分析项缺少微博ID")
        wid = str(item["微博ID"])
        if wid in found:
            raise InvalidResponse("返回了重复微博ID")
        found[wid] = item
    if set(found) != wanted:
        raise InvalidResponse("返回的微博ID与请求批次不一致")
    output = {}
    for post in posts:
        wid = post["wid"]
        item = found[wid]
        sentiment = item.get("综合情感")
        if not isinstance(sentiment, dict) or not isinstance(sentiment.get("类型"), str) or not sentiment["类型"].strip():
            raise InvalidResponse(f"微博 {wid} 缺少情感类型")
        intensity = sentiment.get("强度")
        if isinstance(intensity, bool) or not isinstance(intensity, (int, float)) or not 0 <= intensity <= 10:
            raise InvalidResponse(f"微博 {wid} 的情感强度无效")
        if not isinstance(sentiment.get("核心触发词", []), list):
            raise InvalidResponse(f"微博 {wid} 的关键词无效")
        aspect = item.get("方面级情感", {})
        threads = item.get("跨微博关联", [])
        if not isinstance(aspect, dict) or not isinstance(threads, list):
            raise InvalidResponse(f"微博 {wid} 的可选字段类型无效")
        output[wid] = {
            "created_at": post["_time"].strftime("%Y-%m-%d %H:%M:%S"),
            "post_interval": post["post_interval"],
            "mentions": post["mentions"],
            "topics": post["topics"],
            "comprehensive_sentiment": {
                "type": sentiment["类型"],
                "intensity": intensity,
                "event_summary": sentiment.get("事件概述", ""),
                "event_category": sentiment.get("事件类型", ""),
                "keywords": sentiment.get("核心触发词", []),
            },
            "aspect_sentiment": aspect,
            "narrative_threads": threads,
        }
    return output


def _read_state(path: Path) -> dict[tuple[str, str], dict]:
    latest = {}
    if not path.exists():
        return latest
    with path.open("rb+") as stream:
        data = stream.read()
        if data and not data.endswith(b"\n"):
            last_newline = data.rfind(b"\n")
            stream.truncate(last_newline + 1)
            data = data[:last_newline + 1]
            print("已忽略并截断 state.jsonl 末尾的不完整记录")
    for number, line in enumerate(data.splitlines(), 1):
        try:
            event = json.loads(line)
            if event["status"] not in {"succeeded", "content_rejected", "retryable_error", "invalid_response"}:
                raise ValueError("未知状态")
            latest[(event["uid"], event["batch_id"])] = event
        except (KeyError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"state.jsonl:{number}: {error}") from error
    return latest


def _append_state(path: Path, latest: dict, event: dict) -> None:
    event["at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    with path.open("ab") as stream:
        stream.write((json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    latest[(event["uid"], event["batch_id"])] = event


@contextlib.contextmanager
def _run_lock(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
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


def _result_path(run_dir: Path, uid: str, batch_id: str) -> Path:
    return run_dir / "results" / uid / f"{batch_id}.json"


def _verified_result(path: Path, uid: str, batch_id: str, wids: list[str], expected_hash: str | None = None) -> str | None:
    if not path.exists():
        return None
    raw = path.read_bytes()
    digest = _digest_bytes(raw)
    if expected_hash is not None and digest != expected_hash:
        raise ValueError(f"结果文件摘要不匹配：{path}")
    try:
        result = json.loads(raw)
        if result["uid"] != uid or result["batch_id"] != batch_id or result["wids"] != wids or set(result["items"]) != set(wids):
            raise ValueError("微博ID覆盖不完整")
        if any(not item.get("comprehensive_sentiment", {}).get("type") for item in result["items"].values()):
            raise ValueError("存在空情感结果")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"结果文件格式无效：{path}: {error}") from error
    return digest


def _process_batch(
    uid: str, batch_id: str, posts: list[dict], run_dir: Path, latest: dict,
    client: Callable[[list[dict], str], dict], max_retries: int, retry_delay: float,
) -> tuple[int, int, int]:
    """Return succeeded, excluded, and unresolved post counts."""
    wids = [post["wid"] for post in posts]
    key = (uid, batch_id)
    old = latest.get(key)
    path = _result_path(run_dir, uid, batch_id)
    if old and (old.get("wids") != wids):
        raise ValueError(f"批次 {uid}/{batch_id} 的微博ID与已有状态不一致")
    if old and old["status"] == "succeeded":
        if _verified_result(path, uid, batch_id, wids, old.get("result_sha256")) is None:
            raise ValueError(f"状态记录成功但缺少结果：{path}")
        return len(posts), 0, 0
    if old and old["status"] == "content_rejected":
        if len(posts) == 1:
            return 0, 1, 0
        middle = len(posts) // 2
        a = _process_batch(uid, batch_id + ".1", posts[:middle], run_dir, latest, client, max_retries, retry_delay)
        b = _process_batch(uid, batch_id + ".2", posts[middle:], run_dir, latest, client, max_retries, retry_delay)
        return tuple(x + y for x, y in zip(a, b))
    if not old and path.exists():
        digest = _verified_result(path, uid, batch_id, wids)
        _append_state(run_dir / "state.jsonl", latest, {
            "uid": uid, "batch_id": batch_id, "wids": wids, "status": "succeeded",
            "result_sha256": digest, "recovered": True,
        })
        return len(posts), 0, 0

    payload = _input_payload(posts)
    for attempt in range(1, max_retries + 1):
        try:
            items = format_response(posts, client(payload, PROMPT))
            _atomic_json(path, {"uid": uid, "batch_id": batch_id, "wids": wids, "items": items})
            digest = _verified_result(path, uid, batch_id, wids)
            _append_state(run_dir / "state.jsonl", latest, {
                "uid": uid, "batch_id": batch_id, "wids": wids, "status": "succeeded",
                "result_sha256": digest, "attempts": attempt,
            })
            return len(posts), 0, 0
        except ContentRejected as error:
            _append_state(run_dir / "state.jsonl", latest, {
                "uid": uid, "batch_id": batch_id, "wids": wids, "status": "content_rejected",
                "scope": "post" if len(posts) == 1 else "batch",
                "error_code": error.code, "request_id": error.request_id, "attempts": attempt,
            })
            if len(posts) == 1:
                return 0, 1, 0
            middle = len(posts) // 2
            a = _process_batch(uid, batch_id + ".1", posts[:middle], run_dir, latest, client, max_retries, retry_delay)
            b = _process_batch(uid, batch_id + ".2", posts[middle:], run_dir, latest, client, max_retries, retry_delay)
            return tuple(x + y for x, y in zip(a, b))
        except Exception as error:
            status = "invalid_response" if isinstance(error, InvalidResponse) else "retryable_error"
            if attempt == max_retries:
                _append_state(run_dir / "state.jsonl", latest, {
                    "uid": uid, "batch_id": batch_id, "wids": wids, "status": status,
                    "error_type": type(error).__name__, "error_code": _provider_code(error) or None,
                    "http_status": getattr(error, "status_code", None),
                    "request_id": getattr(error, "request_id", None), "attempts": attempt,
                })
                return 0, 0, len(posts)
            time.sleep(retry_delay)
    raise AssertionError("unreachable")


def run_analysis(
    input_dir: Path, run_dir: Path, model: str, base_url: str,
    client: Callable[[list[dict], str], dict], max_users: int | None = None,
    max_retries: int = 3, retry_delay: float = 5,
) -> dict:
    if "deepseek" not in model.lower():
        raise ValueError("当前分析程序只支持 DeepSeek 模型")
    if max_retries < 1 or retry_delay < 0 or (max_users is not None and max_users < 1):
        raise ValueError("重试次数、延迟或用户上限无效")
    with _run_lock(run_dir):
        users, offsets, dataset = scan_dataset(input_dir)
        identity = {
            "schema_version": SCHEMA_VERSION,
            "input_dir": str(input_dir.resolve()),
            **dataset,
            "model": model,
            "base_url": base_url,
            "prompt_sha256": _digest_bytes(PROMPT.encode("utf-8")),
            "batch_size": BATCH_SIZE,
            "max_tokens": 4096,
            "temperature": 1.0,
        }
        run_path = run_dir / "run.json"
        if run_path.exists():
            if json.loads(run_path.read_text(encoding="utf-8")) != identity:
                raise ValueError("输入、模型或提示词与已有实验不一致，请使用新的实验目录")
        else:
            _atomic_json(run_path, identity)
        latest = _read_state(run_dir / "state.jsonl")
        report = {"users": 0, "complete": 0, "complete_with_exclusions": 0, "partial": 0,
                  "succeeded_posts": 0, "excluded_posts": 0, "unresolved_posts": 0}
        with (input_dir / "weibo_cleaned.jsonl").open("rb") as stream:
            for uid in users[:max_users]:
                posts = load_user_posts(stream, offsets[uid])
                counts = [0, 0, 0]
                for number, start in enumerate(range(0, len(posts), BATCH_SIZE), 1):
                    outcome = _process_batch(uid, str(number), posts[start:start + BATCH_SIZE], run_dir, latest, client, max_retries, retry_delay)
                    counts = [x + y for x, y in zip(counts, outcome)]
                report["users"] += 1
                report["succeeded_posts"] += counts[0]
                report["excluded_posts"] += counts[1]
                report["unresolved_posts"] += counts[2]
                if counts[2]:
                    report["partial"] += 1
                elif counts[1]:
                    report["complete_with_exclusions"] += 1
                else:
                    report["complete"] += 1
                if report["users"] % 20 == 0:
                    print(f"已处理 {report['users']} 位用户，完整 {report['complete']}，有排除项 {report['complete_with_exclusions']}，待重试 {report['partial']}", flush=True)
        return report


def _provider_code(error: Exception) -> str:
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        nested = body.get("error", body)
        if isinstance(nested, dict) and nested.get("code"):
            return str(nested["code"])
    return str(getattr(error, "code", "") or "")


class DeepSeekClient:
    def __init__(self, model: str, base_url: str, api_key: str):
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("真实模型调用需要安装 openai") from error
        self.model = model
        # Keep retries in this runner so each attempt is counted in state.jsonl.
        self.client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=180.0)

    def __call__(self, batch: list[dict], prompt: str) -> dict:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt},
                          {"role": "user", "content": json.dumps(batch, ensure_ascii=False)}],
                response_format={"type": "json_object"},
                max_tokens=4096,
                temperature=1.0,
                stream=False,
            )
        except Exception as error:
            code = _provider_code(error)
            marker = (code + " " + str(error)).lower()
            if any(term in marker for term in ("data_inspection_failed", "datainspectionfailed", "content_filter", "high risk", "content moderation")):
                raise ContentRejected(code or "content_filter", getattr(error, "request_id", None)) from error
            raise
        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) == "content_filter" or getattr(choice.message, "refusal", None):
            raise ContentRejected("content_filter", getattr(response, "id", None))
        output = choice.message.content
        if not isinstance(output, str):
            raise InvalidResponse("模型未返回文本")
        output = output.strip()
        if output.startswith("```json") and output.endswith("```"):
            output = output[7:-3].strip()
        try:
            return json.loads(output)
        except json.JSONDecodeError as error:
            raise InvalidResponse("模型返回的内容不是 JSON") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "run"))
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--run-id", help="新实验的唯一名称，仅 run 命令需要")
    parser.add_argument("--config", type=Path, help="本地模型配置 JSON，仅 run 命令需要")
    parser.add_argument("--max-users", type=int, help="只处理输入中的前 N 位用户，便于试运行")
    args = parser.parse_args()
    if args.command == "inspect":
        _, _, report = scan_dataset(args.input_dir)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if not args.run_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_id) or not args.config:
        parser.error("run 需要 --run-id 和 --config；run-id 只能包含字母、数字、下划线及连字符")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model, base_url, key_env = (config.get(key) for key in ("model", "base_url", "api_key_env"))
    if not all(isinstance(value, str) and value.strip() for value in (model, base_url, key_env)):
        parser.error("配置需包含非空的 model、base_url、api_key_env")
    api_key = os.environ.get(key_env)
    if not api_key:
        parser.error(f"环境变量 {key_env} 未设置")
    client = DeepSeekClient(model, base_url, api_key)
    report = run_analysis(args.input_dir, DEFAULT_RUNS / args.run_id, model, base_url, client, args.max_users)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
