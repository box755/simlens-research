"""Stage A step (a): Whisper-Large-v3 transcription with timestamps.

Spec: SimLens_Research_Plan_v4.1.md §3.2 Step 1.2 (a).
Runs on the 5090; uses faster-whisper for ~3x speed over the original.

Language handling (decided 2026-05-05):
  - We do NOT force language="en" — that triggers Whisper's translation mode
    on non-English audio, which silently produces a translated transcript and
    hides the fact that the video is not English.
  - Instead we let Whisper auto-detect language + record it. Downstream we run
    filter_non_english.py to drop videos with detected language != "en".
  - This is option C from the 2026-05-05 sanity-check discussion: keep it simple,
    detect first, filter later.

Output: data/whisper/{video_id}.json
  {
    "video_id": ...,
    "duration": 132.5,
    "language": "en",                # detected, NOT forced
    "language_probability": 0.99,    # Whisper's confidence
    "segments": [{"start": 0.5, "end": 2.1, "text": "Hi everyone"}, ...]
  }

Usage:
  python scripts/stage_a/run_whisper.py \
    --manifest data/raw_videos/manifest.jsonl \
    --videos-dir data/raw_videos \
    --out-dir data/whisper
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def get_model(model_size: str):
    """Cache-loaded WhisperModel (one process, one model)."""
    if not hasattr(get_model, "_cache"):
        get_model._cache = {}
    if model_size not in get_model._cache:
        from faster_whisper import WhisperModel  # type: ignore
        get_model._cache[model_size] = WhisperModel(
            model_size, device="cuda", compute_type="float16"
        )
    return get_model._cache[model_size]


def transcribe(audio_path: Path, model_size: str = "large-v3"):
    model = get_model(model_size)
    segments, info = model.transcribe(
        str(audio_path),
        word_timestamps=False,
        language=None,        # auto-detect — see module docstring
        vad_filter=True,
    )
    seg_list = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]
    return info.duration, info.language, info.language_probability, seg_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--ext", default="mp4")
    ap.add_argument("--force", action="store_true",
                    help="re-transcribe even if output exists")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = Path(args.videos_dir)

    with Path(args.manifest).open() as f:
        videos = [json.loads(l) for l in f]
    print(f"Transcribing {len(videos)} videos with Whisper {args.model}")

    n_done = n_skip = n_missing = 0
    languages: dict[str, int] = {}
    for v in videos:
        out_path = out_dir / f"{v['id']}.json"
        if out_path.exists() and not args.force:
            n_skip += 1
            continue
        audio = videos_dir / f"{v['id']}.{args.ext}"
        if not audio.exists():
            n_missing += 1
            continue
        print(f"  transcribing {v['id']}...", flush=True)
        duration, language, language_prob, segs = transcribe(audio, args.model)
        out_path.write_text(
            json.dumps({
                "video_id": v["id"],
                "duration": duration,
                "language": language,
                "language_probability": language_prob,
                "segments": segs,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        languages[language] = languages.get(language, 0) + 1
        n_done += 1
        print(f"    -> lang={language} ({language_prob:.2f}) dur={duration:.1f}s segs={len(segs)}",
              flush=True)
    print()
    print(f"Summary: done={n_done} skipped={n_skip} missing={n_missing}")
    if languages:
        print("Language distribution:")
        for lang, count in sorted(languages.items(), key=lambda x: -x[1]):
            print(f"  {lang}: {count}")


if __name__ == "__main__":
    main()
