"""Stage B Phase 1: distill Claude into 800 sparse JSON lists (100 videos x 8 personas).

Spec: SimLens_Research_Plan_v4.1.md §3.2 Step 1.3.
Cost: 800 calls × ~$0.015 = ~$12 USD.

Pre-flight (Day 1 of Week 3): run on 5 videos × 8 personas = 40 calls (~$0.6) and
hand-check ≥80% (timestamp, comment) pairs before scaling to 100. Use --limit 5
for the sanity check.

Usage:
  python scripts/stage_b_phase1/distill_claude.py \
    --manifest data/raw_videos/manifest.jsonl \
    --personas data/personas/personas_with_activity.yaml \
    --timelines-dir data/timeline_scripts \
    --out data/distillation/claude_outputs.jsonl \
    [--limit 5]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.schema import schema_compliant, timestamp_valid


PROMPT = """You are a YouTube viewer with this persona:
{persona_description}

Expected comment frequency on a typical 2-minute video: {low}-{high} comments.

You just finished watching a short video. Below is the complete timeline of what happened:

{timeline_script}

Reflect on the entire video. List the moments where you would have left a comment, staying in character with the persona description above. Match your persona's expected comment count range above (scaled to the actual video duration).

Output ONLY a valid JSON array in this exact format:
[
  {{"timestamp": "MM:SS", "comment": "your comment here"}},
  ...
]

If nothing struck you, output an empty array: [].

Important:
- Choose timestamps that fall within actual segment ranges shown above
- Each comment should be 10-50 words
- This is post-hoc reflection (you finished watching)"""


def parse_yaml_personas(text: str) -> list[dict]:
    out: list[dict] = []
    cur: dict | None = None
    desc_lines: list[str] | None = None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        m = re.match(r"^(P\d+):\s*$", raw)
        if m:
            if cur:
                if desc_lines is not None:
                    cur["description"] = "\n".join(desc_lines).rstrip()
                out.append(cur)
            cur = {"id": m.group(1)}
            desc_lines = None
            continue
        if cur is None:
            continue
        if raw.startswith("  description:"):
            desc_lines = []
        elif raw.startswith("  expected_comment_count_range:"):
            if desc_lines is not None:
                cur["description"] = "\n".join(desc_lines).rstrip()
                desc_lines = None
            rest = raw.split(":", 1)[1].strip()
            rest = rest.split("#")[0].strip()
            cur["expected_comment_count_range"] = json.loads(rest)
        elif raw.startswith("    ") and desc_lines is not None:
            desc_lines.append(raw[4:])
    if cur:
        if desc_lines is not None:
            cur["description"] = "\n".join(desc_lines).rstrip()
        out.append(cur)
    return out


def call_claude(prompt: str, model: str) -> str:
    from anthropic import Anthropic  # type: ignore

    client = Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def already_done(out_path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not out_path.exists():
        return done
    with out_path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                done.add((rec["video_id"], rec["persona_id"]))
            except json.JSONDecodeError:
                continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--personas", required=True)
    ap.add_argument("--timelines-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all videos; otherwise process first N")
    ap.add_argument("--model", default="claude-sonnet-4-5-20250929",
                    help="snapshot version for reproducibility; matches v4.1 §3.1 Teacher")
    args = ap.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(2)

    with Path(args.manifest).open() as f:
        videos = [json.loads(l) for l in f]
    if args.limit:
        videos = videos[: args.limit]

    personas = parse_yaml_personas(Path(args.personas).read_text())
    print(f"Distilling: {len(videos)} videos × {len(personas)} personas = {len(videos)*len(personas)} calls")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = already_done(out_path)
    print(f"  resuming with {len(done)} already-done pairs")

    n_ok = 0
    n_scr_fail = 0
    n_tvr_fail = 0
    n_total = 0
    with out_path.open("a") as out_f:
        for v in videos:
            ts_path = Path(args.timelines_dir) / f"{v['id']}.txt"
            if not ts_path.exists():
                print(f"  WARN: missing timeline for {v['id']}, skip")
                continue
            timeline = ts_path.read_text()
            duration = float(v.get("duration_sec") or 120)
            for p in personas:
                key = (v["id"], p["id"])
                if key in done:
                    continue
                lo, hi = p["expected_comment_count_range"]
                prompt = PROMPT.format(
                    persona_description=p["description"],
                    low=lo,
                    high=hi,
                    timeline_script=timeline,
                )
                try:
                    raw = call_claude(prompt, args.model)
                except Exception as e:
                    print(f"  ERROR {v['id']}/{p['id']}: {e}")
                    continue
                n_total += 1
                ok, parsed, scr_reason = schema_compliant(raw)
                tvr_ok, tvr_reason = (False, "scr_failed")
                if ok:
                    tvr_ok, tvr_reason = timestamp_valid(parsed, duration)
                rec = {
                    "video_id": v["id"],
                    "persona_id": p["id"],
                    "duration_sec": duration,
                    "raw_output": raw,
                    "scr_ok": ok,
                    "scr_reason": scr_reason,
                    "tvr_ok": tvr_ok,
                    "tvr_reason": tvr_reason,
                    "parsed": parsed if ok else None,
                }
                out_f.write(json.dumps(rec) + "\n")
                out_f.flush()
                if ok and tvr_ok:
                    n_ok += 1
                elif not ok:
                    n_scr_fail += 1
                else:
                    n_tvr_fail += 1
                n = len(parsed) if parsed else 0
                print(f"  {v['id']}/{p['id']}: SCR={ok} TVR={tvr_ok} n={n}")
    if n_total:
        print(f"\nDistillation summary:")
        print(f"  total new calls: {n_total}")
        print(f"  fully ok (SCR+TVR): {n_ok} ({n_ok/n_total*100:.1f}%)")
        print(f"  SCR fail: {n_scr_fail}")
        print(f"  TVR fail (SCR ok): {n_tvr_fail}")
    else:
        print("\nAll pairs already complete (resumed).")


if __name__ == "__main__":
    main()
