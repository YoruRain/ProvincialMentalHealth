"""Focused checks for profile selection and first-pass dynamics metrics."""

from __future__ import annotations

import math
import shutil
import sys
import unittest
import uuid
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "04_result_analysis" / "04_2_emotion_dynamics_profile"))
from build_profiles import build_profiles  # noqa: E402


class EmotionDynamicsProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        data_root = Path(__file__).resolve().parents[1] / "data"
        self.root = data_root / f"test-emotion-profile-{uuid.uuid4().hex}"
        self.root.mkdir()
        self.addCleanup(self._cleanup, data_root)
        self.input = self.root / "analysis_table.parquet"
        self.profile = self.root / "profile.parquet"
        self.audit = self.root / "selection.parquet"

    def _cleanup(self, data_root: Path) -> None:
        if self.root.resolve().is_relative_to(data_root.resolve()) and self.root.name.startswith("test-emotion-profile-"):
            shutil.rmtree(self.root)

    def test_selection_and_transition_denominators(self) -> None:
        rows = [
            # Input order differs from time and post-ID order. User a has a zero-second gap.
            ("a", "2", "2025-01-01", 0, 0),
            ("a", "1", "2025-01-01", -1, -8),
            ("a", "3", "2025-02-01", 1, 6),
            ("b", "1", "2025-01-01", 1, 5),
            ("b", "2", "2025-01-16", 1, 5),
            ("b", "3", "2025-02-01", 1, 5),
            ("c", "1", "2025-01-01", -1, -5),
            ("c", "2", "2025-02-01", None, None),
            ("c", "3", "2025-03-01", 1, 5),
            ("d", "1", "2025-01-01", 1, 5),
            ("d", "2", "2025-01-02", 1, 5),
        ]
        pd.DataFrame([
            {"user_id": uid, "post_id": wid, "province": "湖南", "gender": "f",
             "created_at": pd.Timestamp(date), "valence": valence, "signed_score": score}
            for uid, wid, date, valence, score in rows
        ]).to_parquet(self.input, index=False)

        result = build_profiles(self.input, self.profile, self.audit, min_posts=3, min_span_days=30, min_active_days=2)
        profiles = pd.read_parquet(self.profile).set_index("user_id")
        audit = pd.read_parquet(self.audit).set_index("user_id")

        self.assertEqual((result["users_total"], result["users_included"]), (4, 2))
        self.assertEqual((profiles.loc["a", "n_active_days"], profiles.loc["a", "max_gap_days"]), (2, 31))
        self.assertEqual(set(profiles.index), {"a", "b"})
        self.assertEqual((profiles.loc["a", "n_origin_neg"], profiles.loc["a", "n_origin_neu"], profiles.loc["a", "n_origin_pos"]), (1, 1, 0))
        self.assertEqual((profiles.loc["a", "p_neg_to_neu"], profiles.loc["a", "p_neu_to_pos"]), (1.0, 1.0))
        self.assertTrue(math.isnan(profiles.loc["a", "p_pos_to_pos"]))
        self.assertAlmostEqual(profiles.loc["a", "rmssd_score"], math.sqrt(50))
        self.assertAlmostEqual(profiles.loc["a", "mean_score"], -2 / 3)
        self.assertEqual(profiles.loc["a", "overall_stay_rate"], 0)
        self.assertEqual((profiles.loc["b", "sd_score"], profiles.loc["b", "rmssd_score"], profiles.loc["b", "p_pos_to_pos"]), (0, 0, 1))
        self.assertTrue(math.isnan(profiles.loc["b", "p_neg_to_neg"]))
        self.assertEqual(audit.loc["c", "excluded_reason"], "missing_score")
        self.assertEqual(audit.loc["d", "excluded_reason"], "too_few_posts;short_span")

        stricter_profile = self.root / "stricter_profile.parquet"
        stricter_audit = self.root / "stricter_audit.parquet"
        stricter = build_profiles(
            self.input, stricter_profile, stricter_audit,
            min_posts=3, min_span_days=30, min_active_days=3,
        )
        self.assertEqual(stricter["users_included"], 1)
        self.assertEqual(pd.read_parquet(stricter_profile)["user_id"].tolist(), ["b"])
        self.assertEqual(
            pd.read_parquet(stricter_audit).set_index("user_id").loc["a", "excluded_reason"],
            "too_few_active_days",
        )


if __name__ == "__main__":
    unittest.main()
