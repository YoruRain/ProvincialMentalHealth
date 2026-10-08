"""Focused checks for the legacy long-form analysis table."""

from __future__ import annotations

import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path

import pyarrow.parquet as pq


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "04_result_analysis" / "04_1_data_analysis_table"))
from build_table import build_table  # noqa: E402


class AnalysisTableTests(unittest.TestCase):
    def setUp(self) -> None:
        data_root = Path(__file__).resolve().parents[1] / "data"
        self.root = data_root / f"test-analysis-table-{uuid.uuid4().hex}"
        self.root.mkdir()
        self.addCleanup(self._cleanup, data_root)
        self.output = self.root / "result.parquet"

    def _cleanup(self, data_root: Path) -> None:
        if self.root.resolve().is_relative_to(data_root.resolve()) and self.root.name.startswith("test-analysis-table-"):
            shutil.rmtree(self.root)

    def write_jsonl(self, name: str, rows: list[dict]) -> None:
        with (self.root / name).open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def test_join_nulls_and_tie_order(self) -> None:
        self.write_jsonl("user_cleaned.jsonl", [
            {"uid": "u1", "IP": "湖南", "gender": "f"},
            {"uid": "u2", "IP": "江苏", "gender": "m"},
        ])
        self.write_jsonl("weibo_cleaned.jsonl", [
            {"uid": "u1", "wid": "2", "created_at": "2025-01-01T00:00:00", "attitudes_count": "2", "comments_count": "1", "reposts_count": "0"},
            {"uid": "u1", "wid": "1", "created_at": "2025-01-01T00:00:00", "attitudes_count": "3", "comments_count": "0", "reposts_count": "1"},
            {"uid": "u1", "wid": "3", "created_at": "2025-01-01T00:01:00", "attitudes_count": "0", "comments_count": "0", "reposts_count": "0"},
            {"uid": "u1", "wid": "4", "created_at": "2025-01-01T00:02:00", "attitudes_count": "0", "comments_count": "0", "reposts_count": "0"},
            {"uid": "u1", "wid": "5", "created_at": "2025-01-01T00:03:00", "attitudes_count": "0", "comments_count": "0", "reposts_count": "0"},
            {"uid": "u2", "wid": "6", "created_at": "2025-01-01T00:00:00", "attitudes_count": "0", "comments_count": "0", "reposts_count": "0"},
        ])
        results = {"u1": {
            "1": {"comprehensive_sentiment": {"type": "消极-焦虑", "intensity": 8, "event_category": "健康", "event_summary": "生病"}},
            "2": {"comprehensive_sentiment": {"type": "中立", "intensity": 3}},
            "3": {"comprehensive_sentiment": {"type": "混合-复杂", "intensity": 7, "event_category": "工作"}},
            "4": {"comprehensive_sentiment": {}},
            "5": {"comprehensive_sentiment": {"type": "中性-观察", "intensity": 4}},
        }}
        analysis_path = self.root / "llm_analysis.json"
        analysis_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")

        report = build_table(self.root, analysis_path, self.output)
        rows = pq.read_table(self.output).to_pylist()
        self.assertEqual(report, {"rows": 6, "analyzed_rows": 5, "missing_analysis_rows": 1})
        self.assertEqual([row["post_id"] for row in rows], ["1", "2", "3", "4", "5", "6"])
        self.assertEqual((rows[0]["sentiment_category_raw"], rows[0]["sentiment_subcategory_raw"], rows[0]["signed_score"]), ("消极", "焦虑", -8))
        self.assertEqual((rows[0]["event_category_raw"], rows[0]["event_category"]), ("健康", None))
        self.assertEqual((rows[1]["emotion"], rows[1]["delta_t"], rows[1]["prev_score"], rows[1]["delta_score"]), (None, 0, -8, 8))
        self.assertEqual((rows[2]["sentiment_category_raw"], rows[2]["valence"], rows[2]["prev_score"], rows[2]["delta_score"]), ("混合", None, 0, None))
        self.assertIsNone(rows[3]["intensity"])
        self.assertEqual((rows[4]["valence"], rows[4]["signed_score"], rows[4]["delta_t"]), (0, 0, 60))
        self.assertEqual((rows[5]["province"], rows[5]["has_llm_analysis"], rows[5]["signed_score"]), ("江苏", False, None))
        self.assertEqual((rows[0]["likes"], rows[0]["comments"], rows[0]["reposts"]), (3, 0, 1))


if __name__ == "__main__":
    unittest.main()
