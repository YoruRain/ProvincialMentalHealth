"""Build first-pass emotion dynamics profiles from the long-form Parquet table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_NEW = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = PROJECT_NEW / "data" / "llm_analysis" / "analysis_table.parquet"
DEFAULT_PROFILE = PROJECT_NEW / "data" / "llm_analysis" / "emotion_dynamics_profile_v1.parquet"
DEFAULT_AUDIT = PROJECT_NEW / "data" / "llm_analysis" / "emotion_dynamics_selection_v1.parquet"

INPUT_COLUMNS = ["user_id", "post_id", "province", "gender", "created_at", "valence", "signed_score"]
STATES = [("neg", -1), ("neu", 0), ("pos", 1)]
TRANSITION_COLUMNS = [f"p_{start}_to_{end}" for start, _ in STATES for end, _ in STATES]
PROFILE_COLUMNS = [
    "user_id", "province", "gender", "n_posts", "n_active_days", "span_days",
    "median_gap_days", "max_gap_days",
    "mean_score", "sd_score", "rmssd_score", "overall_stay_rate",
    "n_origin_neg", "n_origin_neu", "n_origin_pos", *TRANSITION_COLUMNS,
]
AUDIT_COLUMNS = [
    "user_id", "n_posts", "n_missing_scores", "n_active_days", "span_days",
    "included", "excluded_reason",
]


def write_parquet_atomically(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        frame.to_parquet(temporary, index=False, engine="pyarrow")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build_profiles(
    input_path: Path,
    profile_path: Path,
    audit_path: Path,
    min_posts: int = 20,
    min_span_days: float = 30,
    min_active_days: int = 10,
) -> dict:
    if min_posts < 2 or min_span_days < 0 or min_active_days < 2:
        raise ValueError("min_posts and min_active_days must be >= 2; min_span_days must be >= 0")
    if profile_path.resolve() == audit_path.resolve():
        raise ValueError("profile and audit output paths must differ")

    data = pd.read_parquet(input_path, columns=INPUT_COLUMNS)
    if data.duplicated(["user_id", "post_id"]).any():
        raise ValueError("duplicate user/post IDs in analysis table")
    if data["created_at"].isna().any():
        raise ValueError("analysis table contains missing timestamps")
    data = data.sort_values(["user_id", "created_at", "post_id"], kind="stable")

    profiles = []
    audit = []
    for uid, posts in data.groupby("user_id", sort=False):
        n = len(posts)
        n_missing = int(posts["signed_score"].isna().sum())
        n_active_days = int(posts["created_at"].dt.normalize().nunique())
        span_days = (posts["created_at"].iloc[-1] - posts["created_at"].iloc[0]).total_seconds() / 86_400
        reasons = []
        if n_missing:
            reasons.append("missing_score")
        if n < min_posts:
            reasons.append("too_few_posts")
        if span_days < min_span_days:
            reasons.append("short_span")
        if n_active_days < min_active_days:
            reasons.append("too_few_active_days")
        audit.append({
            "user_id": uid,
            "n_posts": n,
            "n_missing_scores": n_missing,
            "n_active_days": n_active_days,
            "span_days": span_days,
            "included": not reasons,
            "excluded_reason": ";".join(reasons) if reasons else None,
        })
        if reasons:
            continue

        scores = posts["signed_score"].to_numpy(dtype=np.float64)
        valence = posts["valence"].to_numpy(dtype=np.int8)
        if not np.isin(valence, [-1, 0, 1]).all():
            raise ValueError(f"user {uid} has an invalid valence")
        counts = np.zeros((3, 3), dtype=np.int64)
        np.add.at(counts, (valence[:-1] + 1, valence[1:] + 1), 1)
        origins = counts.sum(axis=1)
        gaps = np.diff(posts["created_at"].to_numpy(dtype="datetime64[ns]")) / np.timedelta64(1, "D")
        profile = {
            "user_id": uid,
            "province": posts["province"].iloc[0],
            "gender": posts["gender"].iloc[0],
            "n_posts": n,
            "n_active_days": n_active_days,
            "span_days": span_days,
            "median_gap_days": float(np.median(gaps)),
            "max_gap_days": float(np.max(gaps)),
            "mean_score": float(np.mean(scores)),
            "sd_score": float(np.std(scores, ddof=1)),
            "rmssd_score": float(np.sqrt(np.mean(np.square(np.diff(scores))))),
            "overall_stay_rate": float(np.trace(counts) / (n - 1)),
        }
        for row, (start, _) in enumerate(STATES):
            profile[f"n_origin_{start}"] = int(origins[row])
            for column, (end, _) in enumerate(STATES):
                profile[f"p_{start}_to_{end}"] = float(counts[row, column] / origins[row]) if origins[row] else None
        profiles.append(profile)

    write_parquet_atomically(pd.DataFrame(profiles, columns=PROFILE_COLUMNS), profile_path)
    write_parquet_atomically(pd.DataFrame(audit, columns=AUDIT_COLUMNS), audit_path)
    return {
        "users_total": len(audit),
        "users_included": len(profiles),
        "users_excluded": len(audit) - len(profiles),
        "min_posts": min_posts,
        "min_span_days": min_span_days,
        "min_active_days": min_active_days,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--profile-output", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--min-posts", type=int, default=20)
    parser.add_argument("--min-span-days", type=float, default=30)
    parser.add_argument("--min-active-days", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(build_profiles(
        args.input, args.profile_output, args.audit_output,
        args.min_posts, args.min_span_days, args.min_active_days,
    ), ensure_ascii=False))


if __name__ == "__main__":
    main()
