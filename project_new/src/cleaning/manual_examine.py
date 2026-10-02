"""Review locally cleaned users that need a human decision."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from cleaning import sample_weibos
from storage import iter_states, load_state, now_utc, save_state


def review_user(data_root: Path, uid: str, decision: str) -> dict:
    """Persist one decision immediately; ``1`` keeps the user, ``0`` rejects it."""
    if decision not in ("0", "1"):
        raise ValueError("人工复核只能选择 1（保留）或 0（拒绝）")
    state = load_state(data_root, uid)
    if state is None or state.get("status") != "needs_review":
        raise ValueError(f"用户 {uid} 不在待人工复核状态")
    state["status"] = "manual_accepted" if decision == "1" else "manual_rejected"
    state["reason"] = "manual_review"
    state["reviewed_at"] = now_utc()
    state["updated_at"] = state["reviewed_at"]
    save_state(data_root, state)
    return state


def _source_unchanged(data_root: Path, state: dict) -> bool:
    directory = data_root / "crawler_results" / str(state["uid"])
    files = list(directory.glob("*.json"))
    return len(files) == 1 and hashlib.sha256(files[0].read_bytes()).hexdigest() == state.get("source_sha256")


def run_review(data_root: Path, selected_uid: str | None = None) -> int:
    pending = [state for state in iter_states(data_root) if state.get("status") == "needs_review"]
    if selected_uid is not None:
        pending = [state for state in pending if str(state["uid"]) == selected_uid]
    print(f"待人工复核：{len(pending)} 人")
    reviewed = 0
    for state in pending:
        uid = str(state["uid"])
        if not _source_unchanged(data_root, state):
            print(f"用户 {uid} 的爬取文件已变化；请先重新清洗，暂不复核。")
            continue
        print(f"\n用户 {uid} 的微博样本：")
        for number, text in enumerate(sample_weibos(state["weibos"], max_length=100), 1):
            print(f"{number}. {text}")
        while True:
            answer = input("保留 1 / 拒绝 0 / 跳过 s / 退出 q：").strip().lower()
            if answer in ("0", "1", "s", "q"):
                break
            print("请输入 1、0、s 或 q。")
        if answer == "q":
            break
        if answer == "s":
            continue
        review_user(data_root, uid, answer)
        reviewed += 1
        print(f"用户 {uid} 的决定已保存。")
    return reviewed


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=project_root / "data")
    parser.add_argument("--uid", help="只复核指定用户 ID")
    args = parser.parse_args()
    count = run_review(args.data_root.resolve(), selected_uid=args.uid)
    print(f"本次保存了 {count} 个复核决定。需要汇总 JSONL 时运行 cleaning.py export。")


if __name__ == "__main__":
    main()
