"""Extract keywords from each video's Timeline Script + aggregate across all videos.

Spec: SimLens_Research_Plan_v4.1.md §2.3 Step 2 + Slide 13 (centroid query):
  per-video: GPT-4o-mini → 5-10 keywords from LLaVA captions + Whisper transcript
  aggregate: union of unique keywords → centroid query for PersonaChat MMR

Two modes:
  --from-titles   uses title+channel from manifest.jsonl as a cheap proxy
                  (use this for Week 1 sanity check before full Stage A is ready)
  --from-timelines reads data/timeline_scripts/*.txt produced by Stage A
                  (real path; needs Stage A pipeline to have run)

Usage:
  python scripts/persona/extract_keywords.py \
    --manifest data/raw_videos/manifest.jsonl \
    --mode from-titles \
    --out data/personas/aggregated_keywords.txt
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

import re
from collections import Counter


PROMPT = """Extract 5-10 keywords (concrete nouns, brands, activities, themes) from this YouTube video text.

Output ONLY a JSON array of lowercase strings, no commentary.

Text:
{text}
"""


def call_gpt(text: str, model: str = "gpt-4o-mini") -> list[str]:
    from openai import OpenAI  # type: ignore

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT.format(text=text[:4000])}],
        max_tokens=200,
        temperature=0.0,
    )
    raw = resp.choices[0].message.content.strip()
    m = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        return [str(k).lower().strip() for k in json.loads(m.group(0))]
    except json.JSONDecodeError:
        return []


def from_titles_mode(manifest_path: Path) -> list[tuple[str, str]]:
    """Cheap proxy: use title + channel + category as the source text."""
    out: list[tuple[str, str]] = []
    with manifest_path.open() as f:
        for line in f:
            v = json.loads(line)
            text = f"Title: {v['title']}\nChannel: {v['channel']}\nCategory: {v['category']}"
            out.append((v["id"], text))
    return out


def from_timelines_mode(manifest_path: Path, timelines_dir: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    with manifest_path.open() as f:
        for line in f:
            v = json.loads(line)
            ts_path = timelines_dir / f"{v['id']}.txt"
            if ts_path.exists():
                out.append((v["id"], ts_path.read_text()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--mode", choices=["from-titles", "from-timelines"], required=True)
    ap.add_argument("--timelines-dir", default="data/timeline_scripts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--per-video-cache", default="data/personas/per_video_keywords.jsonl")
    args = ap.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(2)

    manifest = Path(args.manifest)
    if args.mode == "from-titles":
        items = from_titles_mode(manifest)
    else:
        items = from_timelines_mode(manifest, Path(args.timelines_dir))
    print(f"Extracting keywords from {len(items)} videos via {args.model}")

    cache_path = Path(args.per_video_cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    if cache_path.exists():
        with cache_path.open() as f:
            for line in f:
                seen.add(json.loads(line)["video_id"])
        print(f"  skipping {len(seen)} videos already in cache")

    counter: Counter[str] = Counter()
    with cache_path.open("a") as cache_fp:
        for vid, text in items:
            if vid in seen:
                with cache_path.open() as rf:
                    for line in rf:
                        rec = json.loads(line)
                        if rec["video_id"] == vid:
                            counter.update(rec["keywords"])
                            break
                continue
            kws = call_gpt(text, model=args.model)
            print(f"  {vid}: {kws}")
            counter.update(kws)
            cache_fp.write(json.dumps({"video_id": vid, "keywords": kws}) + "\n")

    aggregated = " ".join(k for k, _ in counter.most_common())
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(aggregated)
    print(f"\nWrote {len(counter)} unique keywords to {out_path}")
    print("Top 20 keywords:")
    for k, c in counter.most_common(20):
        print(f"  {c:3d}  {k}")


if __name__ == "__main__":
    main()
