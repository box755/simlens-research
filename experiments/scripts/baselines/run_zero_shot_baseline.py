"""Zero-shot sparse JSON commentary baselines for Table 1a.

Spec: SimLens_Research_Plan_v4.1.md §5.1 + §5.4 Table 1a + §8.2 (Batch + cache).
Three baselines, all consume the SAME Timeline Script + persona description as
SimLens, output the SAME sparse JSON schema:

  - claude    → claude-sonnet-4-5-20250929  (Teacher; reused from Phase 1)
  - openai    → gpt-4o-mini                 (independent LLM family baseline)
  - llama-zs  → meta-llama/Llama-3.2-3B-Instruct, no LoRA (untrained floor)

Recommended workflow:
  1. Sanity check (sync, --limit 5) — verify prompt + parsing on small subset
  2. Full run (batch) — submit 800 calls via Anthropic / OpenAI Batches API
     For Claude: piggy-back distill_claude_batch.py prompt-caching strategy

Cost (v4.1 budget, after Batch + 1h cache):
  - claude:   800 calls ≈ $3.5  (vs sync no-cache $12)
  - openai:   800 calls ≈ $2    (vs sync $4)
  - llama-zs: $0 (local 5090 inference)

Usage:
  # Sanity check
  python scripts/baselines/run_zero_shot_baseline.py \
    --backend openai --limit 5 \
    --manifest data/raw_videos/manifest.jsonl \
    --personas data/personas/personas_with_activity.yaml \
    --timelines-dir data/timeline_scripts \
    --out data/baselines/openai_sanity.jsonl

  # Full Llama zero-shot (800 calls, ~30 min on 5090)
  python scripts/baselines/run_zero_shot_baseline.py --backend llama-zs \
    ... --out data/baselines/llama_zs_outputs.jsonl

For full Claude / OpenAI baselines at scale, use the dedicated batch scripts:
  - scripts/stage_b_phase1/distill_claude_batch.py    (Anthropic batch + cache)
  - scripts/baselines/openai_batch.py                 (OpenAI batch; TODO)
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

from stage_b_phase1.distill_claude import PROMPT, parse_yaml_personas
from utils.schema import schema_compliant, timestamp_valid


def call_claude(prompt: str, model: str) -> str:
    from anthropic import Anthropic  # type: ignore

    client = Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def call_openai(prompt: str, model: str) -> str:
    from openai import OpenAI  # type: ignore

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.7,
    )
    return resp.choices[0].message.content


def call_llama_zero_shot(prompt: str, model_id: str) -> str:
    """Llama-3.2-3B without any LoRA — the 'untrained floor' baseline."""
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    import torch  # type: ignore

    if not hasattr(call_llama_zero_shot, "_model"):
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        call_llama_zero_shot._model = (model, tok)
    model, tok = call_llama_zero_shot._model
    msgs = [{"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True).to(model.device)
    out = model.generate(inputs, max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.95)
    return tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["claude", "openai", "llama-zs"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--personas", required=True)
    ap.add_argument("--timelines-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--claude-model", default="claude-sonnet-4-5-20250929")
    ap.add_argument("--openai-model", default="gpt-4o-mini")
    ap.add_argument("--llama-model", default="meta-llama/Llama-3.2-3B-Instruct")
    args = ap.parse_args()

    if args.backend == "claude" and "ANTHROPIC_API_KEY" not in os.environ:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(2)
    if args.backend == "openai" and "OPENAI_API_KEY" not in os.environ:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(2)

    with Path(args.manifest).open() as f:
        videos = [json.loads(l) for l in f]
    if args.limit:
        videos = videos[: args.limit]
    personas = parse_yaml_personas(Path(args.personas).read_text())
    print(f"[{args.backend}] {len(videos)}×{len(personas)} = {len(videos)*len(personas)} pairs")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, str]] = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add((rec["video_id"], rec["persona_id"]))
                except json.JSONDecodeError:
                    pass
        print(f"  resuming: {len(done)} pairs already done")

    n_ok = n_scr_fail = n_tvr_fail = 0
    with out_path.open("a") as out_f:
        for v in videos:
            ts_path = Path(args.timelines_dir) / f"{v['id']}.txt"
            if not ts_path.exists():
                continue
            timeline = ts_path.read_text()
            duration = float(v.get("duration_sec") or 120)
            for p in personas:
                if (v["id"], p["id"]) in done:
                    continue
                lo, hi = p["expected_comment_count_range"]
                prompt = PROMPT.format(
                    persona_description=p["description"],
                    low=lo, high=hi,
                    timeline_script=timeline,
                )
                try:
                    if args.backend == "claude":
                        raw = call_claude(prompt, args.claude_model)
                    elif args.backend == "openai":
                        raw = call_openai(prompt, args.openai_model)
                    else:
                        raw = call_llama_zero_shot(prompt, args.llama_model)
                except Exception as e:
                    print(f"  ERROR {v['id']}/{p['id']}: {e}")
                    continue
                ok, parsed, scr_reason = schema_compliant(raw)
                tvr_ok, tvr_reason = (False, "scr_failed")
                if ok:
                    tvr_ok, tvr_reason = timestamp_valid(parsed, duration)
                rec = {
                    "video_id": v["id"], "persona_id": p["id"], "backend": args.backend,
                    "duration_sec": duration, "raw_output": raw,
                    "scr_ok": ok, "scr_reason": scr_reason,
                    "tvr_ok": tvr_ok, "tvr_reason": tvr_reason,
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
                print(f"  {v['id']}/{p['id']}: SCR={ok} TVR={tvr_ok}")
    total = n_ok + n_scr_fail + n_tvr_fail
    if total:
        print(f"\n[{args.backend}] summary: ok={n_ok}/{total}  scr_fail={n_scr_fail}  tvr_fail={n_tvr_fail}")


if __name__ == "__main__":
    main()
