"""Offline checks for exporting a completed LLM run to result analysis."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "02_llm_analysis"))
from aggregate import aggregate_run  # noqa: E402


class AggregateRunTests(unittest.TestCase):
    def setUp(self) -> None:
        data_root = Path(__file__).resolve().parents[1] / "data"
        self.root = data_root / f"test-llm-aggregate-{uuid.uuid4().hex}"
        if not self.root.resolve().is_relative_to(data_root.resolve()):
            raise RuntimeError("测试目录不在工作区内")
        self.root.mkdir()
        self.addCleanup(self._cleanup, data_root)
        self.input_dir = self.root / "input"
        self.input_dir.mkdir()
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()

        users = [{"uid": "101"}, {"uid": "202"}]
        posts = [
            {"uid": "101", "wid": "3", "created_at": "2026-04-03T12:00:00"},
            {"uid": "202", "wid": "4", "created_at": "2026-04-01T12:00:00"},
            {"uid": "101", "wid": "2", "created_at": "2026-04-02T12:00:00"},
            {"uid": "101", "wid": "1", "created_at": "2026-04-01T12:00:00"},
        ]
        for name, rows in (("user", users), ("weibo", posts)):
            path = self.input_dir / f"{name}_cleaned.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        run = {"schema_version": 1, "input_dir": str(self.input_dir.resolve()),
               "user_sha256": self.digest(self.input_dir / "user_cleaned.jsonl"),
               "weibo_sha256": self.digest(self.input_dir / "weibo_cleaned.jsonl"),
               "users": 2, "weibos": 4, "batch_size": 2}
        (self.run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
        self.events: list[dict] = []

    def _cleanup(self, data_root: Path) -> None:
        if self.root.resolve().is_relative_to(data_root.resolve()) and self.root.name.startswith("test-llm-aggregate-"):
            shutil.rmtree(self.root)

    @staticmethod
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def add_success(self, uid: str, batch_id: str, wids: list[str]) -> None:
        path = self.run_dir / "results" / uid / f"{batch_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        items = {wid: {"comprehensive_sentiment": {"type": "中立-平静", "intensity": 5}}
                 for wid in wids}
        path.write_text(json.dumps({"uid": uid, "batch_id": batch_id,
                                    "wids": wids, "items": items}), encoding="utf-8")
        self.events.append({"uid": uid, "batch_id": batch_id, "wids": wids,
                            "status": "succeeded", "result_sha256": self.digest(path)})

    def add_rejection(self, uid: str, batch_id: str, wids: list[str]) -> None:
        self.events.append({"uid": uid, "batch_id": batch_id, "wids": wids,
                            "status": "content_rejected", "scope": "post" if len(wids) == 1 else "batch"})

    def save_events(self) -> None:
        (self.run_dir / "state.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in self.events), encoding="utf-8")

    def test_split_rejection_and_success_export(self) -> None:
        self.add_rejection("101", "1", ["1", "2"])
        self.add_success("101", "1.1", ["1"])
        self.add_rejection("101", "1.2", ["2"])
        self.add_success("101", "2", ["3"])
        self.add_success("202", "1", ["4"])
        self.save_events()

        report = aggregate_run(self.run_dir)
        output = json.loads((self.run_dir / "llm_analysis.json").read_text(encoding="utf-8"))
        self.assertEqual((report["users"], report["succeeded_posts"], report["excluded_posts"]), (2, 3, 1))
        self.assertEqual(set(output), {"101", "202"})
        self.assertEqual(set(output["101"]), {"1", "3"})
        self.assertEqual(set(output["202"]), {"4"})

    def test_unresolved_batch_preserves_previous_export(self) -> None:
        self.add_success("101", "1", ["1", "2"])
        self.events.append({"uid": "101", "batch_id": "2", "wids": ["3"], "status": "retryable_error"})
        self.add_success("202", "1", ["4"])
        self.save_events()
        output = self.run_dir / "llm_analysis.json"
        output.write_text("previous export", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "retryable_error"):
            aggregate_run(self.run_dir)
        self.assertEqual(output.read_text(encoding="utf-8"), "previous export")

    def test_changed_input_or_result_hash_blocks_export(self) -> None:
        self.add_success("101", "1", ["1", "2"])
        self.add_success("101", "2", ["3"])
        self.add_success("202", "1", ["4"])
        self.save_events()
        result = self.run_dir / "results" / "101" / "1.json"
        result.write_text(result.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "摘要不匹配"):
            aggregate_run(self.run_dir)

        self.add_success("101", "1", ["1", "2"])
        self.save_events()
        with (self.input_dir / "weibo_cleaned.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "输入摘要"):
            aggregate_run(self.run_dir)

    def test_incomplete_state_record_is_not_truncated(self) -> None:
        self.add_success("101", "1", ["1", "2"])
        self.add_success("101", "2", ["3"])
        self.add_success("202", "1", ["4"])
        self.save_events()
        state_path = self.run_dir / "state.jsonl"
        with state_path.open("ab") as stream:
            stream.write(b'{"uid":"101"')
        before = state_path.read_bytes()

        with self.assertRaisesRegex(ValueError, "未完整写入"):
            aggregate_run(self.run_dir)
        self.assertEqual(state_path.read_bytes(), before)

    def test_rejects_unknown_state_and_protects_input_files(self) -> None:
        self.add_success("101", "1", ["1", "2"])
        self.add_success("101", "2", ["3"])
        self.add_success("202", "1", ["4"])
        self.events.append({"uid": "999", "batch_id": "1", "wids": ["9"], "status": "retryable_error"})
        self.save_events()

        with self.assertRaisesRegex(ValueError, "当前输入之外"):
            aggregate_run(self.run_dir)
        self.assertFalse((self.run_dir / "llm_analysis.json").exists())

        self.events.pop()
        self.save_events()
        user_path = self.input_dir / "user_cleaned.jsonl"
        before = user_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "不能覆盖"):
            aggregate_run(self.run_dir, user_path)
        self.assertEqual(user_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
