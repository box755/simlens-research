"""Stage A step (c): assemble Timeline Script from Whisper + LLaVA outputs.

Spec: SimLens_Research_Plan_v4.1.md §3.2 Step 1.2 (c).

Format:
=== Timeline Script ===
[00:00-00:10] Visual: <LLaVA caption>
              Audio: <Whisper transcript chunk>
...
=== End ===

This is pure data assembly — no models needed. Runs locally or on 5090.

Usage:
  python scripts/stage_a/build_timeline_script.py \
    --manifest data/raw_videos/manifest.jsonl \
    --whisper-dir data/whisper \
    --llava-dir data/llava \
    --out-dir data/timeline_scripts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt_ts(sec: float) -> str:
    sec_int = int(sec)
    return f"{sec_int // 60:02d}:{sec_int % 60:02d}"


def assemble(video_id: str, whisper: dict, llava: dict) -> tuple[str, int]:
    """Merge Whisper transcript chunks into LLaVA's segment grid.

    Returns (timeline_text, n_failed_segments).
    """
    lines = [f"=== Timeline Script for video {video_id} (duration {whisper['duration']:.1f}s) ==="]
    n_failed = 0
    for seg in llava["segments"]:
        start, end = seg["start"], seg["end"]
        chunk = " ".join(
            s["text"].strip()
            for s in whisper["segments"]
            if s["end"] >= start and s["start"] < end
        ).strip() or "(silent)"
        cap = seg.get("caption")
        if not cap:
            cap = "(visual unavailable)"
            n_failed += 1
        lines.append(f"[{fmt_ts(start)}-{fmt_ts(end)}] Visual: {cap}")
        lines.append(f"              Audio: {chunk}")
    lines.append("=== End ===")
    return "\n".join(lines), n_failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--whisper-dir", required=True)
    ap.add_argument("--llava-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.manifest).open() as f:
        videos = [json.loads(l) for l in f]

    n_done = 0
    n_skipped = 0
    failed_segs_total = 0
    for v in videos:
        wp = Path(args.whisper_dir) / f"{v['id']}.json"
        lp = Path(args.llava_dir) / f"{v['id']}.json"
        if not wp.exists() or not lp.exists():
            print(f"  skip {v['id']}: missing whisper or llava output")
            n_skipped += 1
            continue
        whisper = json.loads(wp.read_text(encoding="utf-8"))
        llava = json.loads(lp.read_text(encoding="utf-8"))
        text, n_failed = assemble(v["id"], whisper, llava)
        if n_failed:
            print(f"  {v['id']}: {n_failed} segments had unavailable visuals")
            failed_segs_total += n_failed
        (out_dir / f"{v['id']}.txt").write_text(text, encoding="utf-8")
        n_done += 1
    print(f"\nBuilt {n_done} timeline scripts -> {out_dir}")
    if n_skipped:
        print(f"  skipped {n_skipped} videos (missing inputs)")
    if failed_segs_total:
        print(f"  {failed_segs_total} segments across all videos marked '(visual unavailable)'")


if __name__ == "__main__":
    main()
