"""Select the top-100 usable English videos with balanced category distribution.

Refines filter_non_english.py by:
  1. Keeping only videos with detected language == 'en' AND probability >= 0.85
  2. Requiring at least N segments AND M total transcript characters (= real narration)
  3. Balancing categories to 20 per group (same as v4.1 §3.2 Step 1.1)
  4. Within each category, ranking by transcript richness so we keep the most
     content-rich videos (more transcript -> more reaction cues for Claude).

Usage:
  python scripts/stage_a/select_top_videos.py \
    --manifest data/raw_videos/manifest.jsonl \
    --whisper-dir data/whisper \
    --out data/raw_videos/manifest.en.jsonl \
    --per-category 20

Output: data/raw_videos/manifest.en.jsonl  (the canonical "use this" manifest
        for all downstream Stage A LLaVA / Stage B distillation steps).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--whisper-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-category", type=int, default=20)
    ap.add_argument("--min-prob", type=float, default=0.85)
    ap.add_argument("--min-segments", type=int, default=3)
    ap.add_argument("--min-chars", type=int, default=100)
    args = ap.parse_args()

    with Path(args.manifest).open() as f:
        manifest = [json.loads(l) for l in f]
    by_id = {v["id"]: v for v in manifest}
    print(f"Manifest size: {len(manifest)}")

    whisper_dir = Path(args.whisper_dir)
    candidates: dict[str, list[tuple]] = defaultdict(list)  # category -> [(richness, video, lang, prob, n_seg, n_char)]
    rejected: dict[str, int] = defaultdict(int)
    no_whisper = 0

    for v in manifest:
        wp = whisper_dir / f"{v['id']}.json"
        if not wp.exists():
            no_whisper += 1
            continue
        try:
            d = json.loads(wp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            rejected["malformed_json"] += 1
            continue
        lang = d.get("language")
        prob = d.get("language_probability") or 0
        n_segs = len(d["segments"])
        n_chars = sum(len(s["text"]) for s in d["segments"])

        if lang != "en":
            rejected[f"non_en_{lang}"] += 1
            continue
        if prob < args.min_prob:
            rejected["low_prob_en"] += 1
            continue
        if n_segs < args.min_segments:
            rejected["too_few_segments"] += 1
            continue
        if n_chars < args.min_chars:
            rejected["too_little_text"] += 1
            continue

        # Richness score: more chars + more segments = richer narration
        richness = n_chars + n_segs * 5
        candidates[v["category"]].append((richness, v, lang, prob, n_segs, n_chars))

    print(f"\nRejection breakdown:")
    print(f"  no_whisper_output: {no_whisper}")
    for reason, count in sorted(rejected.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")
    print()

    # Pick top N per category by richness
    selected: list[dict] = []
    print(f"Selecting top-{args.per_category} per category:")
    for cat in sorted(candidates):
        ranked = sorted(candidates[cat], key=lambda x: -x[0])
        chosen = ranked[: args.per_category]
        selected.extend(c[1] for c in chosen)
        avail = len(ranked)
        print(f"  {cat:25s} {len(chosen)}/{args.per_category}  (available: {avail})")
        if len(chosen) < args.per_category:
            print(f"    !! short by {args.per_category - len(chosen)} — top up needed for this category")
        # Show richness range
        if chosen:
            print(f"    richness: top={chosen[0][0]}  bottom={chosen[-1][0]}  "
                  f"chars={chosen[0][5]}-{chosen[-1][5]}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for v in selected:
            f.write(json.dumps(v) + "\n")
    print(f"\nWrote {len(selected)} videos to {out_path}")
    if len(selected) < args.per_category * 5:
        print(f"  WARN: target was {args.per_category * 5}, got {len(selected)}")


if __name__ == "__main__":
    main()
