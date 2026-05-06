"""Collect 30 test-set videos for v4.2.1 video-level hold-out.

Spec: SimLens_Research_Plan_v4.2.md §3.2 Step 1.1b (v4.2.1 新增).

Different from collect_youtube_videos.py:
  - Uses fresh query keywords (avoid overlap with the train-set 246 video pool)
  - 6 videos per category × 5 categories = 30 test videos
  - Same captioned + 60-180s + en + view>10K filter as train set

Usage:
  python scripts/data_collection/collect_test_videos.py \
    --out data/raw_videos/manifest.test.jsonl \
    --per-category 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.env import load_env

load_env()

from data_collection.collect_youtube_videos import (
    parse_iso_duration,
    search_videos,
    get_video_details,
    filter_video,
)


# Fresh query keywords — different from collect_youtube_videos.py to expand
# the candidate pool beyond the train-set 246-video pool.
TEST_CATEGORIES = {
    "Vlog/Lifestyle": [
        "what i eat in a day",
        "weekly routine",
        "study with me vlog",
        "minimalist lifestyle",
    ],
    "Tech Review": [
        "earbuds review",
        "smart watch test",
        "camera comparison",
        "monitor review",
        "keyboard review",
        "tech tips",
    ],
    "Food/Cooking": [
        "5 minute recipe",
        "viral food trend",
        "breakfast ideas",
        "snack recipe",
    ],
    "Education/How-to": [
        "science explained",
        "history facts",
        "did you know",
        "quick tutorial",
    ],
    "Entertainment/Comedy": [
        "stand up clip",
        "improv comedy",
        "viral moment",
        "celebrity interview",
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-category", type=int, default=6)
    ap.add_argument("--max-search-results", type=int, default=50)
    ap.add_argument("--exclude-manifest", default="data/raw_videos/manifest.jsonl",
                    help="path to existing train manifest to exclude")
    args = ap.parse_args()

    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        print("ERROR: YOUTUBE_API_KEY not set", file=sys.stderr)
        sys.exit(2)

    seen_ids: set[str] = set()
    excl_path = Path(args.exclude_manifest)
    if excl_path.exists():
        with excl_path.open() as f:
            for line in f:
                seen_ids.add(json.loads(line)["id"])
        print(f"Excluding {len(seen_ids)} video IDs from train manifest")

    total_kept: list[dict] = []
    for category, queries in TEST_CATEGORIES.items():
        kept_in_category: list[dict] = []
        for q in queries:
            if len(kept_in_category) >= args.per_category:
                break
            print(f"[{category}] searching: {q!r}")
            try:
                stubs = search_videos(q, api_key, args.max_search_results)
            except Exception as e:
                print(f"  search failed: {e}", file=sys.stderr)
                continue
            ids = [s["id"]["videoId"] for s in stubs if s["id"]["kind"] == "youtube#video"]
            ids = [i for i in ids if i not in seen_ids]
            if not ids:
                continue
            details = get_video_details(ids, api_key)
            for item in details:
                vid = item["id"]
                kept, reason = filter_video(item)
                if not kept:
                    continue
                seen_ids.add(vid)
                kept_in_category.append({
                    "id": vid,
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "category": category,
                    "title": item["snippet"]["title"],
                    "channel": item["snippet"]["channelTitle"],
                    "duration_sec": parse_iso_duration(item["contentDetails"]["duration"]),
                    "views": int(item["statistics"].get("viewCount", 0)),
                    "published_at": item["snippet"].get("publishedAt"),
                    "split": "test",
                })
                if len(kept_in_category) >= args.per_category:
                    break
        print(f"  -> {len(kept_in_category)} kept in {category}")
        total_kept.extend(kept_in_category[: args.per_category])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for v in total_kept:
            f.write(json.dumps(v) + "\n")
    print(f"\nWrote {len(total_kept)} test videos to {out_path}")


if __name__ == "__main__":
    main()
