"""Join legacy cleaned data and LLM results into one row per Weibo post."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_NEW = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = PROJECT_NEW / "data" / "datasets" / "legacy"
DEFAULT_ANALYSIS = DEFAULT_INPUT / "llm_analysis.json"
DEFAULT_OUTPUT = PROJECT_NEW / "data" / "llm_analysis" / "analysis_table.parquet"

SCHEMA = pa.schema(
    [
        ("user_id", pa.string()),
        ("post_id", pa.string()),
        ("province", pa.string()),
        ("gender", pa.string()),
        ("created_at", pa.timestamp("s")),
        ("sentiment_category_raw", pa.string()),
        ("sentiment_subcategory_raw", pa.string()),
        ("valence", pa.int8()),
        ("emotion", pa.string()),
        ("intensity", pa.int8()),
        ("signed_score", pa.int8()),
        ("event_category_raw", pa.string()),
        ("event_category", pa.string()),
        ("event_summary", pa.string()),
        ("delta_t", pa.int64()),
        ("prev_score", pa.int8()),
        ("delta_score", pa.int16()),
        ("likes", pa.int64()),
        ("comments", pa.int64()),
        ("reposts", pa.int64()),
        ("has_llm_analysis", pa.bool_()),
    ]
)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc


def sentiment_fields(analysis: dict | None) -> dict:
    sentiment = (analysis or {}).get("comprehensive_sentiment") or {}
    label = sentiment.get("type")
    category = subcategory = None
    if isinstance(label, str) and label:
        category, _, subcategory = label.partition("-")
        category = category or None
        subcategory = subcategory or None

    valence = {"积极": 1, "消极": -1, "中立": 0, "中性": 0}.get(category)
    intensity = sentiment.get("intensity")
    if not isinstance(intensity, int) or isinstance(intensity, bool) or not 0 <= intensity <= 10:
        intensity = None

    return {
        "sentiment_category_raw": category,
        "sentiment_subcategory_raw": subcategory,
        "valence": valence,
        "emotion": subcategory,
        "intensity": intensity,
        "signed_score": valence * intensity if valence is not None and intensity is not None else None,
        "event_category_raw": sentiment.get("event_category") or None,
        "event_category": None,
        "event_summary": sentiment.get("event_summary") or None,
    }


def build_table(input_dir: Path, analysis_path: Path, output_path: Path) -> dict:
    users = {}
    for user in read_jsonl(input_dir / "user_cleaned.jsonl"):
        uid = str(user["uid"])
        if uid in users:
            raise ValueError(f"duplicate user ID: {uid}")
        users[uid] = user

    posts_by_user = defaultdict(list)
    post_ids = set()
    for post in read_jsonl(input_dir / "weibo_cleaned.jsonl"):
        uid, wid = str(post["uid"]), str(post["wid"])
        if uid not in users:
            raise ValueError(f"post {wid} has no user {uid}")
        if wid in post_ids:
            raise ValueError(f"duplicate post ID: {wid}")
        post_ids.add(wid)
        post["created_at"] = datetime.fromisoformat(post["created_at"])
        posts_by_user[uid].append(post)

    with analysis_path.open("r", encoding="utf-8-sig") as stream:
        results = json.load(stream)
    if not isinstance(results, dict):
        raise ValueError("analysis result must be keyed by user ID")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    writer = None
    batch = []
    analyzed = 0
    rows = 0
    try:
        writer = pq.ParquetWriter(temporary_path, SCHEMA, compression="zstd")
        for uid, posts in posts_by_user.items():
            analyses = results.pop(uid, {})
            if not isinstance(analyses, dict):
                raise ValueError(f"analysis for user {uid} is not an object")
            user = users[uid]
            posts.sort(key=lambda post: (post["created_at"], str(post["wid"])))
            previous_time = previous_score = None
            for post in posts:
                wid = str(post["wid"])
                analysis = analyses.pop(wid, None)
                if analysis is not None and not isinstance(analysis, dict):
                    raise ValueError(f"analysis for post {wid} is not an object")
                fields = sentiment_fields(analysis)
                score = fields["signed_score"]
                timestamp = post["created_at"]
                batch.append(
                    {
                        "user_id": uid,
                        "post_id": wid,
                        "province": user.get("IP") or None,
                        "gender": user.get("gender") or None,
                        "created_at": timestamp,
                        **fields,
                        "delta_t": int((timestamp - previous_time).total_seconds()) if previous_time else None,
                        "prev_score": previous_score,
                        "delta_score": score - previous_score if score is not None and previous_score is not None else None,
                        "likes": int(post["attitudes_count"]),
                        "comments": int(post["comments_count"]),
                        "reposts": int(post["reposts_count"]),
                        "has_llm_analysis": analysis is not None,
                    }
                )
                previous_time, previous_score = timestamp, score
                analyzed += analysis is not None
                rows += 1
                if len(batch) >= 10_000:
                    writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
                    batch.clear()
            if analyses:
                raise ValueError(f"user {uid} has analysis for unknown posts: {list(analyses)[:3]}")
        if results:
            raise ValueError(f"analysis has unknown users: {list(results)[:3]}")
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
        writer.close()
        writer = None
        temporary_path.replace(output_path)
    finally:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)
    return {"rows": rows, "analyzed_rows": analyzed, "missing_analysis_rows": rows - analyzed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build_table(args.input_dir, args.analysis, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
