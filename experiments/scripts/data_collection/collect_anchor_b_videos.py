"""[DEPRECATED 2026-05-06] Collect 50 cross-lingual videos for Bilibili pairing.

⚠️ v4.2.1 砍 Bilibili 採集。本檔留作 future work F-cross-lingual 參考。
**不再用於 SimLens 主 pipeline**。

Original docstring follows:

Spec: SimLens_Research_Plan_v4.2.md §3.2 Step 1.4 Anchor B.

Different from collect_youtube_videos.py:
  1. NO videoCaption filter (we WANT silent / no-dialogue videos)
  2. Query pool targets visually-driven content that translates across languages:
     - music videos / instrumental
     - sports highlights
     - silent comedy / pranks / fail compilations
     - cooking timelapse / satisfying compilation
     - animation / anime no dialogue
  3. Same duration filter (60-180s) and view threshold

Output schema matches collect_youtube_videos.py manifest format so the same
download_videos.py works for these too.

Usage:
  python scripts/data_collection/collect_anchor_b_videos.py \
    --out data/raw_videos/manifest.anchor_b.jsonl \
    --per-category 10 \
    --max-search-results 50 \
    --exclude-manifest data/raw_videos/manifest.jsonl

Targets 5 sub-categories × 10 videos = 50 cross-lingual videos.
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

# Reuse the API call helpers from the main collector
from data_collection.collect_youtube_videos import (
    parse_iso_duration,
    search_videos,
    get_video_details,
    filter_video,
)


CROSS_LINGUAL_CATEGORIES = {
    "Music/Instrumental": [
        "instrumental cover",
        "music video no lyrics",
        "piano cover",
        "guitar performance",
    ],
    "Sports/Highlights": [
        "sports highlights",
        "epic plays compilation",
        "best goals",
        "trick shots",
    ],
    "Silent/Visual Comedy": [
        "silent comedy",
        "fail compilation",
        "prank no talking",
        "funny moments compilation",
    ],
    "Cooking/Satisfying": [
        "cooking timelapse",
        "satisfying compilation",
        "food preparation no talking",
        "asmr cooking",
    ],
    "Animation/Visual Story": [
        "animated short no dialogue",
        "anime fight scene",
        "stop motion short",
        "claymation animation",
    ],
}


def search_videos_no_captions(query: str, api_key: str, max_results: int = 50) -> list[dict]:
    """Like search_videos but does NOT require captions.

    We need silent/visual content, which has no captions. So we drop the
    videoCaption filter (the rest of the params identical).
    """
    import urllib.parse
    import urllib.request

    params = {
        "key": api_key,
        "part": "id,snippet",
        "q": query,
        "type": "video",
        "videoDuration": "short",
        "videoEmbeddable": "true",
        "safeSearch": "strict",
        "relevanceLanguage": "en",
        "maxResults": max_results,
    }
    url = "https://www.googleapis.com/youtube/v3/search?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read())["items"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-category", type=int, default=10)
    ap.add_argument("--max-search-results", type=int, default=50)
    ap.add_argument("--exclude-manifest", default=None,
                    help="path to existing manifest.jsonl whose video_ids to skip")
    args = ap.parse_args()

    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        print("ERROR: YOUTUBE_API_KEY not set", file=sys.stderr)
        sys.exit(2)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_ids: set[str] = set()
    if args.exclude_manifest and Path(args.exclude_manifest).exists():
        with Path(args.exclude_manifest).open() as f:
            for line in f:
                seen_ids.add(json.loads(line)["id"])
        print(f"Excluding {len(seen_ids)} video IDs already in {args.exclude_manifest}")

    total_kept: list[dict] = []
    for category, queries in CROSS_LINGUAL_CATEGORIES.items():
        kept_in_category: list[dict] = []
        for q in queries:
            if len(kept_in_category) >= args.per_category:
                break
            print(f"[{category}] searching: {q!r}")
            try:
                stubs = search_videos_no_captions(q, api_key, args.max_search_results)
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
                    "anchor_b": True,  # mark for downstream filtering
                })
                if len(kept_in_category) >= args.per_category:
                    break
        print(f"  -> {len(kept_in_category)} kept in {category}")
        total_kept.extend(kept_in_category[: args.per_category])

    with out_path.open("w") as f:
        for v in total_kept:
            f.write(json.dumps(v) + "\n")
    print(f"\nWrote {len(total_kept)} cross-lingual videos to {out_path}")


if __name__ == "__main__":
    main()
