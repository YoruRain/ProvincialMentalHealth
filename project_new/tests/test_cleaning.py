"""Focused offline checks for the local cleaning workflow."""

from __future__ import annotations

import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "01_cleaning"))
from cleaning import classify_votes, run_cleaning, vote_with_model  # noqa: E402
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

    def test_missing_ip_manual_review_and_repeatable_export(self) -> None:
        self.write_crawl("101")

        self.assertEqual(run_cleaning(self.data_root, self.config)["needs_review"], 1)
        state = load_state(self.data_root, "101")
        self.assertEqual(state["user"]["IP"], "")
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
        self.assertEqual(list(user_line), ["uid", "screen_name", "gender", "statuses_count", "IP"])
        self.assertEqual(user_line["uid"], "101")
        self.assertEqual(user_line["IP"], "")
        self.assertEqual(state["model_results"], [])
        with (self.data_root / "cleaning_output" / "weibo_cleaned.jsonl").open(encoding="utf-8") as stream:
            posts = [json.loads(line) for line in stream]
        self.assertEqual(len(posts), 12)
        self.assertTrue(all("done" not in post for post in posts))

    def test_short_user_rejected_and_changed_source_preserved(self) -> None:
        path = self.write_crawl("202", count=8, ip_location="湖南（附加说明）")
        self.assertEqual(run_cleaning(self.data_root, self.config)["rejected"], 1)
        self.assertEqual(load_state(self.data_root, "202")["reason"], "too_few_valid_weibos")
        path.write_text(json.dumps(make_crawl("202", count=12, ip_location="湖南"), ensure_ascii=False), encoding="utf-8")
        self.assertEqual(run_cleaning(self.data_root, self.config)["source_changed"], 1)
        self.assertEqual(load_state(self.data_root, "202")["status"], "rejected")
        self.assertEqual(run_cleaning(self.data_root, self.config, reprocess=True)["needs_review"], 1)

    def test_absent_ip_location_is_empty(self) -> None:
        path = self.write_crawl("203")
        payload = json.loads(path.read_text(encoding="utf-8"))
        del payload["user"]["ip_location"]
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        self.assertEqual(run_cleaning(self.data_root, self.config)["needs_review"], 1)
        self.assertEqual(load_state(self.data_root, "203")["user"]["IP"], "")

    def test_three_model_vote_can_accept_without_real_api(self) -> None:
        self.write_crawl("303", ip_location="浙江（说明）")
        self.config.write_text(json.dumps({"models": [
            {"base_url": "https://invalid.example", "model": f"model-{index}", "api_key": "test"}
            for index in range(3)
        ]}), encoding="utf-8")
        results = [{"outcome": "vote", "vote": 1} for _ in range(3)]
        with patch("cleaning.vote_with_model", side_effect=results) as vote:
            self.assertEqual(run_cleaning(self.data_root, self.config)["auto_accepted"], 1)
            self.assertEqual(vote.call_count, 3)
            self.assertEqual(vote.call_args_list[0].args[3][0], "900000")
        state = load_state(self.data_root, "303")
        self.assertEqual(state["votes"], [1, 1, 1])
        self.assertEqual(state["model_results"], results)
        self.assertNotIn("done", state["user"])
        self.assertNotIn("error", state["user"])
        self.assertNotIn("done", state["weibos"][0])
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
        with patch("cleaning.vote_with_model", return_value={"outcome": "error", "message": "offline"}):
            self.assertEqual(run_cleaning(self.data_root, self.config)["error"], 1)
        with patch("cleaning.vote_with_model", side_effect=[
            {"outcome": "vote", "vote": vote} for vote in (1, 1, 0)
        ]):
            self.assertEqual(run_cleaning(self.data_root, self.config)["needs_review"], 1)

    def test_content_filter_is_recorded_and_sent_to_manual_review(self) -> None:
        self.write_crawl("502", ip_location="广东")
        self.config.write_text(json.dumps({"models": [
            {"base_url": "https://invalid.example", "model": f"model-{index}", "api_key": "test"}
            for index in range(3)
        ]}), encoding="utf-8")
        results = [
            {"outcome": "vote", "vote": 1},
            {"outcome": "content_filter", "provider_code": "data_inspection_failed", "sample_wids": ["900000"]},
            {"outcome": "vote", "vote": 1},
        ]
        calls = 0

        def next_result(*args):
            nonlocal calls
            if calls:
                saved = load_state(self.data_root, "502")
                self.assertEqual(saved["status"], "processing")
                self.assertEqual(len(saved["model_results"]), calls)
            result = results[calls]
            calls += 1
            return result

        with patch("cleaning.vote_with_model", side_effect=next_result):
            self.assertEqual(run_cleaning(self.data_root, self.config)["needs_review"], 1)
        state = load_state(self.data_root, "502")
        self.assertEqual(state["reason"], "model_content_filter")
        self.assertEqual(state["votes"], [1, 1])
        self.assertEqual(state["model_results"], results)
        self.assertEqual(run_cleaning(self.data_root, self.config)["skipped_existing"], 1)
        self.assertEqual(export_jsonl(self.data_root)["users"], 0)
        review_user(self.data_root, "502", "1")
        self.assertEqual(load_state(self.data_root, "502")["model_results"], results)
        self.assertEqual(export_jsonl(self.data_root)["users"], 1)

    def test_provider_content_filter_signals_and_generic_400(self) -> None:
        model = {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-plus", "api_key": "test"}
        sample_wids = ["900000"]

        class ProviderError(Exception):
            status_code = 400
            request_id = "request-1"

            def __init__(self, code: str, message: str):
                super().__init__(message)
                self.response = SimpleNamespace(
                    json=lambda: {"error": {"code": code, "message": message}},
                    headers={"x-request-id": "request-1"},
                )

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=None)))
        fake_openai = SimpleNamespace(OpenAI=lambda **kwargs: client)
        with patch.dict(sys.modules, {"openai": fake_openai}), patch("cleaning.time.sleep"):
            client.chat.completions.create = lambda **kwargs: (_ for _ in ()).throw(
                ProviderError("data_inspection_failed", "Input data may contain inappropriate content.")
            )
            blocked = vote_with_model(model, "502", ["正文"], sample_wids)
            self.assertEqual((blocked["outcome"], blocked["attempts"], blocked["request_id"]),
                             ("content_filter", 1, "request-1"))
            client.chat.completions.create = lambda **kwargs: SimpleNamespace(
                choices=[SimpleNamespace(finish_reason="content_filter")], _request_id="request-2"
            )
            deepseek = vote_with_model({**model, "base_url": "https://api.deepseek.com"}, "502", ["正文"], sample_wids)
            self.assertEqual((deepseek["outcome"], deepseek["request_id"]), ("content_filter", "request-2"))
            client.chat.completions.create = lambda **kwargs: (_ for _ in ()).throw(
                ProviderError("content_filter", "The request was rejected because it was considered high risk")
            )
            kimi = vote_with_model({**model, "base_url": "https://api.moonshot.cn/v1"}, "502", ["正文"], sample_wids)
            self.assertEqual((kimi["provider"], kimi["outcome"]), ("kimi", "content_filter"))
            client.chat.completions.create = lambda **kwargs: (_ for _ in ()).throw(
                ProviderError("invalid_request_error", "content_filter: The request was rejected because it was considered high risk")
            )
            kimi_message = vote_with_model({**model, "base_url": "https://api.moonshot.cn/v1"}, "502", ["正文"], sample_wids)
            self.assertEqual(kimi_message["outcome"], "content_filter")
            client.chat.completions.create = lambda **kwargs: (_ for _ in ()).throw(
                ProviderError("invalid_request_error", "Bad parameter")
            )
            generic = vote_with_model(model, "502", ["正文"], sample_wids)
            self.assertEqual((generic["outcome"], generic["attempts"]), ("error", 5))

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
