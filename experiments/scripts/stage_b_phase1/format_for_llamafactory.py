"""Convert Claude distillation outputs to LLaMA-Factory SFT JSONL (one file per persona).

Spec: 8 LoRA adapters → 8 datasets, each only contains that persona's pairs.

Usage:
  python scripts/stage_b_phase1/format_for_llamafactory.py \
    --distill data/distillation/claude_outputs.jsonl \
    --personas data/personas/personas_with_activity.yaml \
    --timelines-dir data/timeline_scripts \
    --out-dir data/distillation/sft

Output (per persona):
  data/distillation/sft/P1.jsonl  with rows
    {"instruction": "...full prompt...", "input": "", "output": "<sparse JSON>"}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage_b_phase1.distill_claude import PROMPT, parse_yaml_personas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distill", required=True)
    ap.add_argument("--personas", required=True)
    ap.add_argument("--timelines-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    personas = {p["id"]: p for p in parse_yaml_personas(Path(args.personas).read_text())}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_persona: dict[str, list[dict]] = {pid: [] for pid in personas}

    with Path(args.distill).open() as f:
        for line in f:
            rec = json.loads(line)
            if not (rec["scr_ok"] and rec["tvr_ok"]):
                continue
            ts = Path(args.timelines_dir) / f"{rec['video_id']}.txt"
            if not ts.exists():
                continue
            persona = personas[rec["persona_id"]]
            lo, hi = persona["expected_comment_count_range"]
            prompt = PROMPT.format(
                persona_description=persona["description"],
                low=lo,
                high=hi,
                timeline_script=ts.read_text(),
            )
            by_persona[rec["persona_id"]].append({
                "instruction": prompt,
                "input": "",
                "output": json.dumps(rec["parsed"], ensure_ascii=False),
            })

    for pid, rows in by_persona.items():
        out_path = out_dir / f"{pid}.jsonl"
        with out_path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {pid}: {len(rows)} pairs -> {out_path}")


if __name__ == "__main__":
    main()
