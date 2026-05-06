"""Collect candidate videos likely to have many YouTube comment timestamp mentions.

Spec: SimLens_Research_Plan_v4.2.1 §3.2 Step 1.1c (2026-05-06 新增).

Strategy:
  1. Search YouTube with "timestamp-friendly" queries (compilations, top-N
     lists, reaction breakdowns, highlights, mixes) — content types where
     viewers commonly write "0:42 best part" / "@1:23" in comments.
  2. For each candidate, pre-fetch 50 top-level comments and count timestamp
     mentions via the same regex used in Anchor B.
  3. Keep only videos with mention >= --min-mentions (default 5).
  4. Up to --target videos collected.
  5. Output appended to manifest.test_v2.jsonl (separate from manifest.test).

Usage:
  python scripts/data_collection/collect_high_mention_videos.py \
    --out data/raw_videos/manifest.test_v2.jsonl \
    --target 30 \
    --min-mentions 5 \
    --exclude-manifest data/raw_videos/manifest.jsonl \
    --exclude-manifest data/raw_videos/manifest.test.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.env import load_env
from data_collection.collect_youtube_videos import (
    parse_iso_duration,
    search_videos,
    get_video_details,
    filter_video,
)
from eval.anchor_metrics import extract_timestamp_mentions

load_env()


# Queries known to attract comments containing timestamp mentions.
# Empirically: compilation/highlight/list/breakdown content >> standard vlog.
HIGH_MENTION_CATEGORIES = {
    "Highlights/Compilation": [
        "highlights compilation",
        "best moments compilation",
        "top 10 moments",
        "epic moments",
        "funniest moments",
    ],
    "Reaction/Breakdown": [
        "reaction breakdown",
        "scene by scene analysis",
        "trailer breakdown",
        "music video reaction",
    ],
    "Sports/Plays": [
        "best plays",
        "incredible goals",
        "trick shots compilation",
        "sports highlights",
    ],
    "Music/Mix": [
        "music mix tracklist",
        "guitar solo compilation",
        "best beat drops",
    ],
    "Tutorial/List": [
        "tips and tricks",
        "step by step tutorial",
        "things you didn't know",
    ],
}


def fetch_comments_count_mentions(
    video_id: str,
    api_key: str,
    duration_sec: float,
    max_pool: int = 200,
) -> tuple[int, int]:
    """Fetch up to max_pool comments (paginated), count timestamp mentions.

    Returns (n_comments_fetched, n_mentions_total).
    """
    n_comments = 0
    n_mentions = 0
    page_token: str | None = None
    while n_comments < max_pool:
        params = {
            "key": api_key,
            "part": "snippet",
            "videoId": video_id,
            "maxResults": min(100, max_pool - n_comments),
            "order": "time",
            "textFormat": "plainText",
        }
        if page_token:
            params["pageToken"] = page_token
        url = "https://www.googleapis.com/youtube/v3/commentThreads?" + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            if "commentsDisabled" in body or e.code == 403:
                return (n_comments, n_mentions)
            return (n_comments, n_mentions)
        except Exception:
            return (n_comments, n_mentions)

        for item in data.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            text = top.get("textDisplay", "").strip()
            if not text:
                continue
            n_comments += 1
            n_mentions += len(extract_timestamp_mentions(text, duration_sec))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return (n_comments, n_mentions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=30,
                    help="how many high-mention videos to collect")
    ap.add_argument("--min-mentions", type=int, default=5,
                    help="minimum timestamp mentions in pre-fetched comments")
    ap.add_argument("--prefetch-comments", type=int, default=200,
                    help="how many top-level comments to pre-fetch per video")
    ap.add_argument("--max-search-results", type=int, default=50)
    ap.add_argument("--exclude-manifest", action="append", default=[],
                    help="paths to existing manifest.jsonl whose video_ids to skip "
                         "(can specify multiple times)")
    ap.add_argument("--sleep", type=float, default=0.3)
    args = ap.parse_args()

    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        print("ERROR: YOUTUBE_API_KEY not set", file=sys.stderr)
        sys.exit(2)

    seen_ids: set[str] = set()
    for excl_path_str in args.exclude_manifest:
        excl_path = Path(excl_path_str)
        if excl_path.exists():
            with excl_path.open() as f:
                for line in f:
                    seen_ids.add(json.loads(line)["id"])
            print(f"Excluding {len(seen_ids)} ids (cumulative) after {excl_path}")

    # Already-collected from this script (resumable)
    out_path = Path(args.out)
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                seen_ids.add(json.loads(line)["id"])

    kept: list[dict] = []
    n_pre_filter_passed = 0
    n_pre_filter_failed = 0
    n_api_calls = 0

    for category, queries in HIGH_MENTION_CATEGORIES.items():
        if len(kept) >= args.target:
            break
        for q in queries:
            if len(kept) >= args.target:
                break
            print(f"[{category}] searching: {q!r}")
            try:
                stubs = search_videos(q, api_key, args.max_search_results)
            except Exception as e:
                print(f"  search failed: {e}", file=sys.stderr)
                continue
            n_api_calls += 1
            ids = [s["id"]["videoId"] for s in stubs if s["id"]["kind"] == "youtube#video"]
            ids = [i for i in ids if i not in seen_ids]
            if not ids:
                continue
            try:
                details = get_video_details(ids, api_key)
                n_api_calls += 1
            except Exception as e:
                print(f"  details failed: {e}", file=sys.stderr)
                continue
            for item in details:
                if len(kept) >= args.target:
                    break
                vid = item["id"]
                kept_filter, reason = filter_video(item)
                if not kept_filter:
                    continue
                duration_sec = parse_iso_duration(item["contentDetails"]["duration"])
                # Pre-filter: count mentions in pre-fetched comments
                n_comments, n_mentions = fetch_comments_count_mentions(
                    vid, api_key, float(duration_sec or 180),
                    max_pool=args.prefetch_comments)
                n_api_calls += 1
                seen_ids.add(vid)
                if n_mentions >= args.min_mentions:
                    n_pre_filter_passed += 1
                    rec = {
                        "id": vid,
                        "url": f"https://www.youtube.com/watch?v={vid}",
                        "category": category,
                        "title": item["snippet"]["title"],
                        "channel": item["snippet"]["channelTitle"],
                        "duration_sec": duration_sec,
                        "views": int(item["statistics"].get("viewCount", 0)),
                        "published_at": item["snippet"].get("publishedAt"),
                        "split": "test",
                        "pre_filter_n_comments": n_comments,
                        "pre_filter_n_mentions": n_mentions,
                    }
                    kept.append(rec)
                    print(f"  KEEP {vid}: {n_mentions} mentions in {n_comments} comments  "
                          f"({len(kept)}/{args.target} kept)")
                    time.sleep(args.sleep)
                else:
                    n_pre_filter_failed += 1
                    print(f"    skip {vid}: only {n_mentions} mentions in {n_comments} comments")
                    time.sleep(args.sleep)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for v in kept:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    print(f"\nSummary:")
    print(f"  kept (high mention): {len(kept)}")
    print(f"  rejected (low mention): {n_pre_filter_failed}")
    print(f"  approx YouTube API calls: {n_api_calls}")
    print(f"  wrote to: {out_path}")


if __name__ == "__main__":
    main()
