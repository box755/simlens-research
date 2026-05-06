"""Collect YouTube comment timestamp mentions for v4.2.1 Anchor B.

Spec: SimLens_Research_Plan_v4.2.md §3.2 Step 1.4 Anchor B (v4.2.1 重寫).

For each video in --manifest:
  1. Pull all top-level comments via commentThreads.list (up to --max-pool)
     using both order=relevance and order=time, dedup by comment_id.
  2. For each comment, regex-extract timestamp mentions ("MM:SS" / "@MM:SS"),
     filter out clock-time false positives (timestamp > video duration + 5s).
  3. Output one JSON per video:
     {
       "video_id": ...,
       "duration_sec": ...,
       "n_comments_pulled": N,
       "n_comments_with_mentions": K,
       "mentions": [
         {"comment_id": ..., "timestamp_sec": 12.3, "comment_text": "..."},
         ...
       ]
     }

Usage:
  python scripts/data_collection/collect_timestamp_mentions.py \
    --manifest data/raw_videos/manifest.test.jsonl \
    --out-dir data/anchor_b/timestamp_mentions \
    --max-pool 200
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
from eval.anchor_metrics import extract_timestamp_mentions

load_env()


def fetch_comments(
    video_id: str,
    api_key: str,
    max_pool: int = 200,
    order: str = "time",
) -> list[dict]:
    """Pull up to max_pool top-level comments from one video."""
    out: list[dict] = []
    page_token: str | None = None
    while len(out) < max_pool:
        params = {
            "key": api_key,
            "part": "snippet",
            "videoId": video_id,
            "maxResults": min(100, max_pool - len(out)),
            "order": order,
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
                return out
            raise
        for item in data.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            text = top.get("textDisplay", "").strip()
            if not text:
                continue
            out.append({
                "comment_id": item["snippet"]["topLevelComment"]["id"],
                "text": text,
                "like_count": int(top.get("likeCount", 0)),
                "published_at": top.get("publishedAt"),
            })
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return out


def fetch_dual_order(video_id: str, api_key: str, max_pool: int) -> list[dict]:
    """Pull from both relevance + time orders, dedup."""
    by_id: dict[str, dict] = {}
    for order in ("time", "relevance"):
        try:
            for c in fetch_comments(video_id, api_key, max_pool, order=order):
                by_id.setdefault(c["comment_id"], c)
        except Exception as e:
            print(f"  WARN order={order}: {type(e).__name__}: {e}")
    return list(by_id.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-pool", type=int, default=200,
                    help="max comments to pull per video per order")
    ap.add_argument("--sleep", type=float, default=0.3,
                    help="seconds between API calls")
    args = ap.parse_args()

    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        print("ERROR: YOUTUBE_API_KEY not set", file=sys.stderr)
        sys.exit(2)

    with Path(args.manifest).open() as f:
        videos = [json.loads(l) for l in f]
    print(f"Collecting timestamp mentions from {len(videos)} videos")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_done = 0
    total_mentions = 0
    for i, v in enumerate(videos, 1):
        out_path = out_dir / f"{v['id']}.json"
        if out_path.exists():
            print(f"  [{i}/{len(videos)}] {v['id']}: skip (exists)")
            n_done += 1
            continue

        comments = fetch_dual_order(v["id"], api_key, args.max_pool)
        duration = float(v.get("duration_sec") or 180)

        mentions: list[dict] = []
        for c in comments:
            ts_list = extract_timestamp_mentions(c["text"], duration)
            for ts in ts_list:
                mentions.append({
                    "comment_id": c["comment_id"],
                    "timestamp_sec": ts,
                    "comment_text": c["text"][:300],  # truncate
                    "like_count": c.get("like_count", 0),
                })

        rec = {
            "video_id": v["id"],
            "title": v.get("title"),
            "category": v.get("category"),
            "duration_sec": duration,
            "n_comments_pulled": len(comments),
            "n_comments_with_mentions": len({m["comment_id"] for m in mentions}),
            "n_mentions_total": len(mentions),
            "mentions": mentions,
        }
        out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        n_done += 1
        total_mentions += len(mentions)
        print(f"  [{i}/{len(videos)}] {v['id']}: {len(comments)} comments → "
              f"{len(mentions)} timestamp mentions")
        time.sleep(args.sleep)

    print(f"\nSummary: {n_done} videos processed, {total_mentions} total mentions")
    if n_done > 0:
        print(f"  avg {total_mentions / n_done:.1f} mentions per video")


if __name__ == "__main__":
    main()
