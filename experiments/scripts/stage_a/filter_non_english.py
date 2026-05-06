"""Filter non-English videos out of the dataset after Whisper auto-detect.

Reads data/whisper/*.json (which have `language` + `language_probability`
recorded by run_whisper.py with auto-detection), classifies each video, and
either:
  - --report      : print summary only (default)
  - --update-manifest : write data/raw_videos/manifest.filtered.jsonl
                        keeping only English videos (probability >= threshold)
  - --remove-files : also delete non-English mp4 + whisper json
                     (only with --i-mean-it)

Decision rule:
  language == "en" AND language_probability >= --min-prob (default 0.85)

Usage:
  # Inspect
  python scripts/stage_a/filter_non_english.py \
    --whisper-dir data/whisper \
    --manifest data/raw_videos/manifest.jsonl \
    --report

  # Generate filtered manifest
  python scripts/stage_a/filter_non_english.py \
    --whisper-dir data/whisper \
    --manifest data/raw_videos/manifest.jsonl \
    --update-manifest data/raw_videos/manifest.en.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def classify(whisper_path: Path, min_prob: float) -> tuple[str, str | None, float | None]:
    """Returns (status, language, prob). status in {ok, non_en, low_prob, missing, malformed}."""
    if not whisper_path.exists():
        return "missing", None, None
    try:
        data = json.loads(whisper_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return f"malformed:{type(e).__name__}", None, None
    lang = data.get("language")
    prob = data.get("language_probability")
    if lang is None:
        return "no_language_field", None, None  # produced by old version of run_whisper.py
    if lang != "en":
        return "non_en", lang, prob
    if prob is not None and prob < min_prob:
        return "low_prob", lang, prob
    return "ok", lang, prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--whisper-dir", required=True)
    ap.add_argument("--min-prob", type=float, default=0.85,
                    help="minimum language_probability to consider 'en' confident")
    ap.add_argument("--report", action="store_true", default=True,
                    help="print summary (default)")
    ap.add_argument("--update-manifest", default=None,
                    help="write filtered manifest to this path")
    args = ap.parse_args()

    with Path(args.manifest).open() as f:
        videos = [json.loads(l) for l in f]
    whisper_dir = Path(args.whisper_dir)

    statuses: dict[str, list[dict]] = {}
    detail_rows: list[tuple] = []
    for v in videos:
        status, lang, prob = classify(whisper_dir / f"{v['id']}.json", args.min_prob)
        statuses.setdefault(status, []).append(v)
        detail_rows.append((v["id"], v["category"], v["duration_sec"], status, lang, prob))

    print(f"=== Whisper language filter (min_prob={args.min_prob}) ===")
    print(f"Total manifest videos: {len(videos)}")
    print(f"With Whisper output:   {sum(1 for r in detail_rows if r[3] != 'missing')}")
    print()
    print("Status breakdown:")
    for status, items in sorted(statuses.items(), key=lambda x: -len(x[1])):
        print(f"  {status}: {len(items)}")

    non_en_rows = [r for r in detail_rows if r[3] == "non_en"]
    if non_en_rows:
        print()
        print(f"Non-English videos ({len(non_en_rows)}):")
        for vid, cat, dur, _, lang, prob in non_en_rows:
            prob_str = f"{prob:.2f}" if prob is not None else "n/a"
            print(f"  {vid}  [{cat:22s}] {dur:>3}s  lang={lang} (prob={prob_str})")

    low_prob_rows = [r for r in detail_rows if r[3] == "low_prob"]
    if low_prob_rows:
        print()
        print(f"Low-confidence English videos ({len(low_prob_rows)}):")
        for vid, cat, dur, _, lang, prob in low_prob_rows:
            print(f"  {vid}  [{cat:22s}] {dur:>3}s  lang={lang} (prob={prob:.2f})")

    if args.update_manifest:
        kept = [v for v, r in zip(videos, detail_rows) if r[3] == "ok"]
        out_path = Path(args.update_manifest)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for v in kept:
                f.write(json.dumps(v) + "\n")
        print(f"\nWrote filtered manifest: {out_path}  ({len(kept)} videos)")


if __name__ == "__main__":
    main()
