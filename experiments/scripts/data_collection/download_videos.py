"""Download 100 YouTube videos in 720p MP4 via yt-dlp.

Spec: SimLens_Research_Plan_v4.1.md §3.2 Step 1.1.
Reads manifest.jsonl produced by collect_youtube_videos.py and downloads each.

Conventions:
  - 720p max (height<=720) — enough for LLaVA frame extraction, saves disk
  - mp4 container (preferred for ffmpeg downstream)
  - resume by default (skip already-downloaded files)
  - rate-limit + retry for resilience

Usage:
  python scripts/data_collection/download_videos.py \
    --manifest data/raw_videos/manifest.jsonl \
    --out-dir data/raw_videos \
    [--limit 5]   # for sanity check first
    [--rate-limit 2M]   # bandwidth cap

Output: data/raw_videos/{video_id}.mp4 (one per video)
Estimated total size: 100 videos × 60-180s × ~1.5 MB/s ≈ 5-15 GB
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def already_downloaded(video_id: str, out_dir: Path) -> Path | None:
    """Return existing .mp4 path if downloaded, else None."""
    candidate = out_dir / f"{video_id}.mp4"
    if candidate.exists() and candidate.stat().st_size > 100 * 1024:  # > 100KB sanity
        return candidate
    # yt-dlp sometimes outputs .mkv etc. — also check those
    for ext in (".mkv", ".webm"):
        alt = out_dir / f"{video_id}{ext}"
        if alt.exists() and alt.stat().st_size > 100 * 1024:
            return alt
    return None


def download_one(video: dict, out_dir: Path, rate_limit: str | None) -> tuple[bool, str]:
    """Download one video. Returns (success, message)."""
    vid = video["id"]
    existing = already_downloaded(vid, out_dir)
    if existing:
        return True, f"skip (exists: {existing.name})"

    # 720p max, prefer mp4, fall back to merge
    fmt = "bv*[height<=720][ext=mp4]+ba[ext=m4a]/bv*[height<=720]+ba/b[height<=720]"
    cmd = [
        "yt-dlp",
        "-f", fmt,
        "--merge-output-format", "mp4",
        "-o", str(out_dir / f"{vid}.%(ext)s"),
        "--no-playlist",
        "--retries", "3",
        "--fragment-retries", "5",
        "--no-warnings",
        "--quiet",
        "--progress",
        video["url"],
    ]
    if rate_limit:
        cmd[1:1] = ["--limit-rate", rate_limit]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip().splitlines()
        return False, f"yt-dlp failed: {err[-1] if err else 'unknown'}"
    if not already_downloaded(vid, out_dir):
        return False, "yt-dlp succeeded but no file produced"
    final = already_downloaded(vid, out_dir)
    size_mb = final.stat().st_size / 1024 / 1024 if final else 0
    return True, f"ok ({final.name}, {size_mb:.1f} MB)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = all videos; otherwise download first N")
    ap.add_argument("--rate-limit", default=None,
                    help="bandwidth cap for yt-dlp, e.g. '2M'; default unlimited")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.manifest).open() as f:
        videos = [json.loads(l) for l in f]
    if args.limit:
        videos = videos[: args.limit]
    print(f"Downloading {len(videos)} videos -> {out_dir}")

    n_ok = n_fail = n_skip = 0
    failures: list[tuple[str, str]] = []
    for i, v in enumerate(videos, 1):
        ok, msg = download_one(v, out_dir, args.rate_limit)
        prefix = f"[{i}/{len(videos)}] {v['id']:15s}"
        if "skip" in msg:
            print(f"{prefix} {msg}")
            n_skip += 1
        elif ok:
            print(f"{prefix} {msg}")
            n_ok += 1
        else:
            print(f"{prefix} FAIL: {msg}")
            n_fail += 1
            failures.append((v["id"], msg))

    print(f"\nSummary: ok={n_ok} skip={n_skip} fail={n_fail}")
    if failures:
        print("\nFailures:")
        for vid, msg in failures:
            print(f"  {vid}: {msg}")
        sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()
