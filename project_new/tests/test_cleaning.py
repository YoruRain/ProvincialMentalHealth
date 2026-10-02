"""Focused offline checks for the local cleaning workflow."""

from __future__ import annotations

import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "cleaning"))
from cleaning import classify_votes, run_cleaning  # noqa: E402
from manual_examine import review_user, run_review  # noqa: E402
from storage import export_jsonl, load_state, validate_export  # noqa: E402


def make_crawl(uid: str, count: int = 12, ip_location: str = "") -> dict:
    return {
        "user": {
            "id": uid, "screen_name": "测试用户", "gender": "f",
            "statuses_count": count, "ip_location": ip_location,
        },
        "weibo": [
            {
                "id": 900000 + number, "user_id": int(uid),
                "text": f"今天记录了一件生活小事 {number}",
                "created_at": "2026-04-15T12:00:00",
                "attitudes_count": 1, "comments_count": 0, "reposts_count": 0,
            }
            for number in range(count)
        ],
    }


class LocalCleaningTests(unittest.TestCase):
    def setUp(self) -> None:
        workspace_data = Path(__file__).resolve().parents[1] / "data"
        self.test_dir = workspace_data / f"test-cleaning-{uuid.uuid4().hex}"
        if not self.test_dir.resolve().is_relative_to(workspace_data.resolve()):
            raise RuntimeError("测试临时目录不在预期工作区内")
        self.test_dir.mkdir()
        self.addCleanup(self.cleanup_test_dir, workspace_data)
        self.data_root = self.test_dir / "data"
        self.config = self.test_dir / "models.json"
        self.config.write_text('{"models": []}', encoding="utf-8")

    def cleanup_test_dir(self, workspace_data: Path) -> None:
        resolved = self.test_dir.resolve()
        if resolved.is_relative_to(workspace_data.resolve()) and resolved.name.startswith("test-cleaning-"):
            shutil.rmtree(resolved)

    def write_crawl(self, uid: str, count: int = 12, ip_location: str = "") -> Path:
        directory = self.data_root / "crawler_results" / uid
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{uid}.json"
        path.write_text(json.dumps(make_crawl(uid, count, ip_location), ensure_ascii=False), encoding="utf-8")
        return path

    def test_fallback_manual_review_and_repeatable_export(self) -> None:
        self.write_crawl("101")
        fallback = self.data_root / "collecting" / "user_id_ip_final.txt"
        fallback.parent.mkdir(parents=True)
        fallback.write_text("101,江苏\n", encoding="utf-8")

        self.assertEqual(run_cleaning(self.data_root, self.config)["needs_review"], 1)
        state = load_state(self.data_root, "101")
        self.assertEqual(state["user"]["IP"], "江苏")
        self.assertEqual(len(state["weibos"]), 12)
        self.assertEqual(run_cleaning(self.data_root, self.config)["skipped_existing"], 1)

        review_user(self.data_root, "101", "1")
        self.assertEqual(load_state(self.data_root, "101")["status"], "manual_accepted")
        first = export_jsonl(self.data_root)
        second = export_jsonl(self.data_root)
        self.assertEqual((first["users"], first["weibos"]), (1, 12))
        self.assertEqual((second["users"], second["weibos"]), (1, 12))
        self.assertEqual(validate_export(self.data_root)["weibos"], 12)
        user_line = json.loads((self.data_root / "cleaning_output" / "user_cleaned.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(list(user_line), ["uid", "screen_name", "gender", "statuses_count", "IP", "done", "error"])
        self.assertEqual((user_line["uid"], user_line["done"], user_line["error"]), ("101", 0, 0))
        with (self.data_root / "cleaning_output" / "weibo_cleaned.jsonl").open(encoding="utf-8") as stream:
            self.assertEqual(sum(1 for _ in stream), 12)

    def test_short_user_rejected_and_changed_source_preserved(self) -> None:
        path = self.write_crawl("202", count=8, ip_location="湖南（附加说明）")
        self.assertEqual(run_cleaning(self.data_root, self.config)["rejected"], 1)
        self.assertEqual(load_state(self.data_root, "202")["reason"], "too_few_valid_weibos")
        path.write_text(json.dumps(make_crawl("202", count=12, ip_location="湖南"), ensure_ascii=False), encoding="utf-8")
        self.assertEqual(run_cleaning(self.data_root, self.config)["source_changed"], 1)
        self.assertEqual(load_state(self.data_root, "202")["status"], "rejected")
        self.assertEqual(run_cleaning(self.data_root, self.config, reprocess=True)["needs_review"], 1)

    def test_three_model_vote_can_accept_without_real_api(self) -> None:
        self.write_crawl("303", ip_location="浙江（说明）")
        self.config.write_text(json.dumps({"models": [
            {"base_url": "https://invalid.example", "model": f"model-{index}", "api_key": "test"}
            for index in range(3)
        ]}), encoding="utf-8")
        with patch("cleaning.vote_with_model", side_effect=[1, 1, 1]) as vote:
            self.assertEqual(run_cleaning(self.data_root, self.config)["auto_accepted"], 1)
            self.assertEqual(vote.call_count, 3)
        self.assertEqual(load_state(self.data_root, "303")["votes"], [1, 1, 1])
        self.assertEqual(classify_votes([1, 1, 0]), "needs_review")
        self.assertEqual(classify_votes([1, 0, 0]), "rejected")

    def test_export_rejects_duplicate_weibo_ids(self) -> None:
        for uid in ("401", "402"):
            self.write_crawl(uid, ip_location="北京")
        run_cleaning(self.data_root, self.config)
        for uid in ("401", "402"):
            review_user(self.data_root, uid, "1")
        with self.assertRaisesRegex(ValueError, "重复或不匹配的微博 ID"):
            export_jsonl(self.data_root)
        self.assertFalse((self.data_root / "cleaning_output" / "manifest.json").exists())

    def test_model_error_state_can_retry(self) -> None:
        self.write_crawl("501", ip_location="广东")
        self.config.write_text(json.dumps({"models": [
            {"base_url": "https://invalid.example", "model": f"model-{index}", "api_key": "test"}
            for index in range(3)
        ]}), encoding="utf-8")
        with patch("cleaning.vote_with_model", side_effect=RuntimeError("offline")):
            self.assertEqual(run_cleaning(self.data_root, self.config)["error"], 1)
        with patch("cleaning.vote_with_model", side_effect=[1, 1, 0]):
            self.assertEqual(run_cleaning(self.data_root, self.config)["needs_review"], 1)

    def test_manual_decision_is_saved_before_quit(self) -> None:
        for uid in ("601", "602"):
            self.write_crawl(uid, ip_location="江苏")
        run_cleaning(self.data_root, self.config)
        with patch("builtins.input", side_effect=["1", "q"]), patch("builtins.print"):
            self.assertEqual(run_review(self.data_root), 1)
        self.assertEqual(load_state(self.data_root, "601")["status"], "manual_accepted")
        self.assertEqual(load_state(self.data_root, "602")["status"], "needs_review")

    def test_nonoriginal_mock_records_are_not_attributed_to_user(self) -> None:
        path = self.write_crawl("701", count=14, ip_location="上海")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["weibo"][0]["retweet"] = {}
        payload["weibo"][1]["user_id"] = 999
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        self.assertEqual(run_cleaning(self.data_root, self.config)["needs_review"], 1)
        state = load_state(self.data_root, "701")
        self.assertEqual((state["skipped_nonoriginal"], len(state["weibos"])), (2, 12))


if __name__ == "__main__":
    unittest.main()
