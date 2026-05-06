"""Collect YouTube comments for v4.2 Anchor A (Real-World Distributional Match).

Spec: SimLens_Research_Plan_v4.2.md §3.2 Step 1.4 Anchor A + §5.2.5.

Per video: pull comments via YouTube Data API v3 commentThreads.list, then
sample 30 to write to disk. Total: 100 × 30 = 3,000 real English comments
used by §5.2.5 Group 1+ deterministic distributional metrics:
  - Length KS-statistic
  - Sentiment Wasserstein distance
  - Style Marker χ²
  - Embedding Frechet Distance

⭐ v4.2 P0c modification (2026-05-05 紀錄.md):
  YouTube API supports order=relevance (default; top comments) and order=time
  (most recent). v4.2 plan said relevance, but that biases toward early /
  high-like / agreeable comments — making SimLens look "more negative /
  informal than real" via sampling artifact, not actual model failure.
  We use order=time and a wider pool to better reflect organic comment
  distribution.

Sampling rule:
  1. Pull up to 100 comments per video, ordered by time (newest first).
  2. Filter: drop very short (<10 chars) and very long (>1000 chars) outliers.
  3. Random sample 30 from the filtered pool (seeded for reproducibility).
  4. If pool < 30 after filter, keep all available.

Usage:
  python scripts/data_collection/collect_youtube_top_comments.py \
    --manifest data/raw_videos/manifest.en.jsonl \
    --out data/anchor_a/youtube_comments.jsonl \
    --per-video 30 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.env import load_env

load_env()


def fetch_comments(
    video_id: str,
    api_key: str,
    max_pool: int = 100,
    order: str = "time",
) -> list[dict]:
    """Fetch up to max_pool top-level comments via commentThreads.list.

    Returns list of {comment_id, text, like_count, published_at, mention_timestamps}.
    """
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
            # 403 commentsDisabled is common; treat as zero comments
            body = e.read().decode("utf-8", errors="ignore")
            if "commentsDisabled" in body or e.code == 403:
                return out
            raise

        for item in data.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            text = top.get("textDisplay", "").strip()
            if not text:
                continue
            mentions = re.findall(r"\b(\d{1,2}:\d{2}(?::\d{2})?)\b", text)
            out.append({
                "comment_id": item["snippet"]["topLevelComment"]["id"],
                "text": text,
                "like_count": int(top.get("likeCount", 0)),
                "published_at": top.get("publishedAt"),
                "mention_timestamps": mentions,
            })
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return out


def filter_pool(pool: list[dict],
                min_chars: int = 10,
                max_chars: int = 1000) -> list[dict]:
    """Drop outliers that distort distribution metrics."""
    return [c for c in pool if min_chars <= len(c["text"]) <= max_chars]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-video", type=int, default=30,
                    help="how many comments to keep per video after filter+sample")
    ap.add_argument("--max-pool", type=int, default=100,
                    help="how many comments to pull from API per video")
    ap.add_argument("--order", default="time",
                    choices=["time", "relevance"],
                    help="YouTube API order param; v4.2 P0c default 'time'")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        print("ERROR: YOUTUBE_API_KEY not set", file=sys.stderr)
        sys.exit(2)

    rng = random.Random(args.seed)

    with Path(args.manifest).open() as f:
        videos = [json.loads(l) for l in f]
    print(f"Collecting top comments for {len(videos)} videos (order={args.order}, "
          f"per-video target={args.per_video})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_videos: set[str] = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_videos.add(rec["video_id"])
                except json.JSONDecodeError:
                    pass
        print(f"  resuming: {len(done_videos)} videos already done")

    n_total_comments = 0
    n_videos_done = 0
    n_videos_skipped = 0
    n_videos_disabled = 0
    with out_path.open("a", encoding="utf-8") as out_f:
        for v in videos:
            if v["id"] in done_videos:
                continue
            try:
                pool = fetch_comments(v["id"], api_key,
                                      max_pool=args.max_pool, order=args.order)
            except Exception as e:
                print(f"  WARN {v['id']}: {type(e).__name__}: {str(e)[:120]}")
                continue
            filtered = filter_pool(pool)
            if not filtered:
                n_videos_disabled += 1
                rec = {
                    "video_id": v["id"], "category": v["category"],
                    "n_pulled": len(pool), "n_filtered": 0,
                    "comments": [],
                    "note": "no comments available (disabled or filtered out)",
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                continue
            if len(filtered) > args.per_video:
                sampled = rng.sample(filtered, args.per_video)
            else:
                sampled = filtered
            rec = {
                "video_id": v["id"], "category": v["category"],
                "n_pulled": len(pool), "n_filtered": len(filtered),
                "n_sampled": len(sampled),
                "comments": sampled,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            n_total_comments += len(sampled)
            n_videos_done += 1
            if (n_videos_done + n_videos_disabled) % 10 == 0:
                print(f"  [{n_videos_done + n_videos_disabled + n_videos_skipped}/"
                      f"{len(videos) - len(done_videos)}] running total: "
                      f"{n_total_comments} comments across {n_videos_done} videos")

    print(f"\nSummary:")
    print(f"  videos processed: {n_videos_done}")
    print(f"  videos with no comments (disabled / filtered out): {n_videos_disabled}")
    print(f"  total comments collected: {n_total_comments}")
    if n_videos_done > 0:
        print(f"  avg comments per video: {n_total_comments / n_videos_done:.1f}")


if __name__ == "__main__":
    main()
