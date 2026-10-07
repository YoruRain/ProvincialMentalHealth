"""Offline checks for the local DeepSeek analysis and its resume protocol."""

from __future__ import annotations

import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "03_llm_analysis"))
from analysis import ContentRejected, run_analysis, scan_dataset  # noqa: E402


class LocalAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        data_root = Path(__file__).resolve().parents[1] / "data"
        self.root = data_root / f"test-llm-analysis-{uuid.uuid4().hex}"
        if not self.root.resolve().is_relative_to(data_root.resolve()):
            raise RuntimeError("测试目录不在工作区内")
        self.root.mkdir()
        self.addCleanup(self._cleanup, data_root)
        self.input_dir = self.root / "input"
        self.input_dir.mkdir()
        self.run_dir = self.root / "run"

    def _cleanup(self, data_root: Path) -> None:
        if self.root.resolve().is_relative_to(data_root.resolve()) and self.root.name.startswith("test-llm-analysis-"):
            shutil.rmtree(self.root)

    def write_input(self, posts: list[tuple[str, str, str]]) -> None:
        users = sorted({uid for uid, _, _ in posts})
        with (self.input_dir / "user_cleaned.jsonl").open("w", encoding="utf-8") as stream:
            for uid in users:
                stream.write(json.dumps({"uid": uid}, ensure_ascii=False) + "\n")
        with (self.input_dir / "weibo_cleaned.jsonl").open("w", encoding="utf-8") as stream:
            for uid, wid, stamp in posts:
                stream.write(json.dumps({"uid": uid, "wid": wid, "text": f"微博 {wid}", "created_at": stamp}, ensure_ascii=False) + "\n")

    @staticmethod
    def response(batch: list[dict], _prompt: str) -> dict:
        return {"analysis": [
            {"微博ID": item["微博ID"], "综合情感": {"类型": "中立-平静", "强度": 5,
             "事件概述": "日常记录", "事件类型": "生活", "核心触发词": ["日常"]},
             "方面级情感": {}, "跨微博关联": []}
            for item in batch
        ]}

    def run_case(self, client, max_users=None) -> dict:
        return run_analysis(self.input_dir, self.run_dir, "deepseek-test", "https://example.invalid", client,
                            max_users=max_users, max_retries=1, retry_delay=0)

    def test_out_of_order_input_resume_and_orphan_result(self) -> None:
        posts = [("101", str(1000 + i), f"2026-04-{(i % 28) + 1:02d}T12:00:00") for i in range(21)]
        posts.insert(3, ("202", "9999", "2026-04-02T12:00:00"))
        self.write_input(list(reversed(posts)))
        calls = []

        def client(batch, prompt):
            calls.append([item["微博ID"] for item in batch])
            return self.response(batch, prompt)

        report = self.run_case(client, max_users=1)
        self.assertEqual((report["complete"], report["succeeded_posts"]), (1, 21))
        self.assertEqual([len(batch) for batch in calls], [20, 1])
        self.assertEqual((calls[0][0], calls[-1][-1]), ("1000", "1020"))
        self.assertEqual(self.run_case(client, max_users=1)["complete"], 1)
        self.assertEqual(len(calls), 2)

        # A crash after writing a result but before its state entry is recoverable.
        state_path = self.run_dir / "state.jsonl"
        events = state_path.read_text(encoding="utf-8").splitlines()
        state_path.write_text(events[0] + "\n", encoding="utf-8")
        self.assertEqual(self.run_case(client, max_users=1)["complete"], 1)
        self.assertEqual(len(calls), 2)
        self.assertTrue(json.loads(state_path.read_text(encoding="utf-8").splitlines()[-1])["recovered"])

        with state_path.open("ab") as stream:
            stream.write(b'{"uid":"incomplete"')
        self.assertEqual(self.run_case(client, max_users=1)["complete"], 1)
        self.assertEqual(len(calls), 2)

        with self.assertRaisesRegex(ValueError, "不一致"):
            run_analysis(self.input_dir, self.run_dir, "deepseek-other", "https://example.invalid", client,
                         max_users=1, max_retries=1, retry_delay=0)

    def test_content_rejection_only_excludes_single_post(self) -> None:
        self.write_input([("101", str(i), "2026-04-01T12:00:00") for i in range(1, 5)])
        calls = []

        def client(batch, prompt):
            calls.append([item["微博ID"] for item in batch])
            if any(item["微博ID"] == "3" for item in batch):
                raise ContentRejected("content_filter")
            return self.response(batch, prompt)

        report = self.run_case(client)
        self.assertEqual((report["complete_with_exclusions"], report["succeeded_posts"], report["excluded_posts"]), (1, 3, 1))
        self.assertGreater(len(calls), 1)
        events = [json.loads(line) for line in (self.run_dir / "state.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([event["wids"] for event in events if event["status"] == "content_rejected" and len(event["wids"]) == 1], [["3"]])
        before = len(calls)
        self.assertEqual(self.run_case(client)["excluded_posts"], 1)
        self.assertEqual(len(calls), before)

    def test_invalid_model_coverage_is_not_success(self) -> None:
        self.write_input([("101", "1", "2026-04-01T12:00:00"), ("101", "2", "2026-04-02T12:00:00")])
        report = self.run_case(lambda batch, prompt: {"analysis": []})
        self.assertEqual((report["partial"], report["unresolved_posts"]), (1, 2))
        self.assertFalse((self.run_dir / "results" / "101" / "1.json").exists())
        self.assertEqual(self.run_case(self.response)["complete"], 1)

    def test_manifest_mismatch_blocks_processing(self) -> None:
        self.write_input([("101", "1", "2026-04-01T12:00:00")])
        (self.input_dir / "manifest.json").write_text('{"user_sha256":"wrong","weibo_sha256":"wrong"}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest.json"):
            scan_dataset(self.input_dir)


if __name__ == "__main__":
    unittest.main()
