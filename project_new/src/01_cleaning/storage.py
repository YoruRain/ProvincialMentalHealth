"""Local state and export files for the cleaning stage."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


USER_FIELDS = ("uid", "screen_name", "gender", "statuses_count", "IP")
WEIBO_FIELDS = (
    "wid", "uid", "text", "created_at", "attitudes_count",
    "comments_count", "reposts_count",
)
ACCEPTED_STATUSES = {"auto_accepted", "manual_accepted"}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def state_path(data_root: Path, uid: str) -> Path:
    if not uid.isdecimal():
        raise ValueError(f"无效用户 ID：{uid!r}")
    return data_root / "cleaning_state" / f"{uid}.json"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_state(data_root: Path, state: dict) -> None:
    path = state_path(data_root, str(state["uid"]))
    _atomic_write(path, json.dumps(state, ensure_ascii=False, indent=2) + "\n")


def load_state(data_root: Path, uid: str) -> dict | None:
    path = state_path(data_root, uid)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as stream:
        state = json.load(stream)
    if str(state.get("uid")) != uid:
        raise ValueError(f"状态文件 ID 与文件名不一致：{path}")
    return state


def iter_states(data_root: Path):
    directory = data_root / "cleaning_state"
    if not directory.exists():
        return
    for path in sorted(directory.glob("*.json")):
        state = load_state(data_root, path.stem)
        if state is not None:
            yield state


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _check_fields(row: dict, expected: tuple[str, ...], source: str) -> None:
    if not isinstance(row, dict) or set(row) != set(expected):
        raise ValueError(f"{source} 的输出字段与清洗格式不一致")


def export_jsonl(data_root: Path) -> dict:
    """Rebuild both JSONL files from accepted user states.

    The state files remain authoritative if a process stops during export.
    The manifest is written last so readers can detect an incomplete pair.
    """
    output_dir = data_root / "cleaning_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    user_path = output_dir / "user_cleaned.jsonl"
    weibo_path = output_dir / "weibo_cleaned.jsonl"
    temporary_paths = []
    try:
        for target in (user_path, weibo_path):
            descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=output_dir)
            temporary_paths.append((target, Path(name), descriptor))
        seen_users: set[str] = set()
        seen_weibos: set[str] = set()
        with os.fdopen(temporary_paths[0][2], "w", encoding="utf-8", newline="\n") as users, \
             os.fdopen(temporary_paths[1][2], "w", encoding="utf-8", newline="\n") as weibos:
            for state in iter_states(data_root):
                if state.get("status") not in ACCEPTED_STATUSES:
                    continue
                uid = str(state["uid"])
                user = state.get("user")
                posts = state.get("weibos")
                _check_fields(user, USER_FIELDS, uid)
                if not isinstance(posts, list) or not posts:
                    raise ValueError(f"用户 {uid} 没有可导出的微博")
                if uid in seen_users or str(user["uid"]) != uid:
                    raise ValueError(f"重复或不匹配的用户 ID：{uid}")
                seen_users.add(uid)
                users.write(json.dumps(user, ensure_ascii=False, separators=(",", ":")) + "\n")
                for post in posts:
                    _check_fields(post, WEIBO_FIELDS, uid)
                    wid = str(post["wid"])
                    if wid in seen_weibos or str(post["uid"]) != uid:
                        raise ValueError(f"重复或不匹配的微博 ID：{wid}")
                    seen_weibos.add(wid)
                    weibos.write(json.dumps(post, ensure_ascii=False, separators=(",", ":")) + "\n")
            for stream in (users, weibos):
                stream.flush()
                os.fsync(stream.fileno())
        for target, temporary_path, _ in temporary_paths:
            os.replace(temporary_path, target)
        manifest = {
            "generated_at": now_utc(),
            "users": len(seen_users),
            "weibos": len(seen_weibos),
            "user_sha256": _digest(user_path),
            "weibo_sha256": _digest(weibo_path),
        }
        _atomic_write(output_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        return manifest
    finally:
        for _, temporary_path, _ in temporary_paths:
            temporary_path.unlink(missing_ok=True)


def validate_export(data_root: Path) -> dict:
    output_dir = data_root / "cleaning_output"
    with (output_dir / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    for filename, key in (("user_cleaned.jsonl", "user_sha256"), ("weibo_cleaned.jsonl", "weibo_sha256")):
        if _digest(output_dir / filename) != manifest[key]:
            raise ValueError(f"导出文件与清单不一致：{filename}；请重新执行 export")
    return manifest
