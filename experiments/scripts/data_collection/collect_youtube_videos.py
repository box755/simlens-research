"""Collect 100 YouTube short videos (1-3min) per genre quota for SimLens.

Spec: SimLens_Research_Plan_v4.1.md §3.2 Step 1.1
  - 5 categories × 20 videos = 100 total
  - 60-180s strict (videoDuration=medium + ISO 8601 second-pass filter)
  - English, public, view > 10K, no 18+, exclude #shorts and 9:16

Usage:
  python scripts/data_collection/collect_youtube_videos.py \
    --out data/raw_videos/manifest.jsonl \
    --per-category 20

Writes a JSONL manifest. Actual video download (yt-dlp) is a separate step:
  yt-dlp -o "data/raw_videos/%(id)s.%(ext)s" -f "bv*[height<=720]+ba/b[height<=720]" \
    --batch-file <(jq -r '.url' data/raw_videos/manifest.jsonl)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.env import load_env

load_env()


CATEGORIES = {
    "Vlog/Lifestyle": ["vlog", "day in my life", "morning routine", "weekly vlog"],
    "Tech Review": ["tech review", "phone unboxing", "laptop review", "gadget"],
    "Food/Cooking": ["recipe", "cooking tutorial", "easy dinner", "food review"],
    "Education/How-to": ["how to", "tutorial", "explained", "learn"],
    "Entertainment/Comedy": ["comedy skit", "funny moments", "reaction", "challenge"],
}

ISO8601_DURATION = re.compile(r"^PT(?:(\d+)M)?(?:(\d+)S)?$")


def parse_iso_duration(s: str) -> int | None:
    """ISO 8601 duration -> total seconds. Returns None if too long (has H) or malformed."""
    if "H" in s:
        return None
    m = ISO8601_DURATION.match(s)
    if not m:
        return None
    minutes = int(m.group(1) or 0)
    seconds = int(m.group(2) or 0)
    return minutes * 60 + seconds


def search_videos(query: str, api_key: str, max_results: int = 50) -> list[dict]:
    """YouTube Data API v3 search.list — returns video stub list."""
    import urllib.parse
    import urllib.request

    params = {
        "key": api_key,
        "part": "id,snippet",
        "q": query,
        "type": "video",
        # YouTube only offers short (<4min) / medium (4-20min) / long (>20min).
        # SimLens needs 60-180s, which lives inside "short". We then filter
        # out the < 60s tail (Shorts) via ISO 8601 duration.
        "videoDuration": "short",
        # Decided 2026-05-05 (Week 2 Day 2): require closed captions = guaranteed
        # narration. Without this, ~30% of harvested videos turn out to be pure
        # music/ASMR/action with no spoken language → Whisper produces empty
        # segments and Timeline Script loses 50% of its reaction cues.
        "videoCaption": "closedCaption",
        "videoEmbeddable": "true",
        "safeSearch": "strict",
        "relevanceLanguage": "en",
        "maxResults": max_results,
    }
    url = "https://www.googleapis.com/youtube/v3/search?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read())["items"]


def get_video_details(video_ids: list[str], api_key: str) -> list[dict]:
    """videos.list — fetch contentDetails (duration), statistics (views), status."""
    import urllib.parse
    import urllib.request

    out: list[dict] = []
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i : i + 50]
        params = {
            "key": api_key,
            "part": "snippet,contentDetails,statistics,status",
            "id": ",".join(chunk),
        }
        url = "https://www.googleapis.com/youtube/v3/videos?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=30) as resp:
            out.extend(json.loads(resp.read())["items"])
    return out


def filter_video(item: dict) -> tuple[bool, str]:
    """Apply SimLens criteria. Returns (kept, reason)."""
    snippet = item.get("snippet", {})
    cd = item.get("contentDetails", {})
    stats = item.get("statistics", {})
    status = item.get("status", {})

    duration_sec = parse_iso_duration(cd.get("duration", ""))
    if duration_sec is None:
        return False, "duration_unparseable_or_too_long"
    if not (60 <= duration_sec <= 180):
        return False, f"duration_out_of_range: {duration_sec}s"

    views = int(stats.get("viewCount", 0))
    if views < 10_000:
        return False, f"low_views: {views}"

    title_desc = (snippet.get("title", "") + " " + snippet.get("description", "")).lower()
    if "#shorts" in title_desc or "#short" in title_desc:
        return False, "shorts_tag"

    if status.get("madeForKids"):
        return False, "made_for_kids"

    if not status.get("embeddable", True):
        return False, "not_embeddable"

    if snippet.get("defaultAudioLanguage") and not snippet["defaultAudioLanguage"].startswith("en"):
        return False, f"non_english: {snippet['defaultAudioLanguage']}"

    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-category", type=int, default=20)
    ap.add_argument("--max-search-results", type=int, default=50,
                    help="how many to pull per query before filtering")
    ap.add_argument("--exclude-manifest", default=None,
                    help="path to existing manifest.jsonl whose video_ids to skip "
                         "(useful for top-up runs)")
    ap.add_argument("--append", action="store_true",
                    help="append to --out instead of overwriting (top-up mode)")
    args = ap.parse_args()

    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        print("ERROR: YOUTUBE_API_KEY not set", file=sys.stderr)
        sys.exit(2)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_kept: list[dict] = []
    seen_ids: set[str] = set()
    if args.exclude_manifest:
        excl = Path(args.exclude_manifest)
        if excl.exists():
            with excl.open() as f:
                for line in f:
                    seen_ids.add(json.loads(line)["id"])
            print(f"Excluding {len(seen_ids)} video IDs already in {excl}")

    for category, queries in CATEGORIES.items():
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
                })
                if len(kept_in_category) >= args.per_category:
                    break
        print(f"  -> {len(kept_in_category)} kept in {category}")
        total_kept.extend(kept_in_category[: args.per_category])

    mode = "a" if args.append else "w"
    with out_path.open(mode) as f:
        for v in total_kept:
            f.write(json.dumps(v) + "\n")
    print(f"\n{'Appended' if args.append else 'Wrote'} {len(total_kept)} videos to {out_path}")


if __name__ == "__main__":
    main()
