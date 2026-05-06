"""Phase 1 distillation via Anthropic Message Batches API + 1h prompt caching.

Spec: SimLens_Research_Plan_v4.1.md §3.2 Step 1.3 + §8.2 (Batch + cache).

Why batch:
  - 50% discount vs sync ($1.50 / M input + $7.50 / M output for Sonnet 4.5)
  - Most batches finish < 1h, max SLA 24h
  - Up to 100k requests / 256MB per batch — 800 calls fits easily

Why 1h prompt caching:
  - For each video, 8 personas share the SAME ~2200-token Timeline Script
  - Mark Timeline Script as cache_control ephemeral with ttl="1h"
  - Persona 1: cache write (1.25× input)
  - Persona 2-8: cache read (0.1× input) — 90% saving on Timeline portion
  - Combined with batch: 87.5% saving over sync-no-cache

Pipeline:
  1. Build 800 request bodies grouped by video (so cache hits stay warm)
  2. POST /v1/messages/batches → batch_id
  3. Poll GET /v1/messages/batches/{id} until processing_status=ended
  4. Stream results from results_url, parse jsonl
  5. Validate SCR + TVR per result, write to output jsonl
  6. Resubmit any expired/failed requests via sync API (small fraction)

Usage:
  python scripts/stage_b_phase1/distill_claude_batch.py \
    --manifest data/raw_videos/manifest.jsonl \
    --personas data/personas/personas_with_activity.yaml \
    --timelines-dir data/timeline_scripts \
    --out data/distillation/claude_outputs.jsonl \
    [--limit 5]   # for sanity check, defaults to all
    [--no-poll]   # submit only and exit (manual poll later via --resume <batch_id>)
    [--resume BATCH_ID]   # poll an existing batch instead of submitting new one
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage_b_phase1.distill_claude import PROMPT, parse_yaml_personas
from utils.schema import schema_compliant, timestamp_valid


# ---------- Prompt assembly with cache_control on Timeline Script ----------

def build_messages_with_cache(persona_desc: str, low: int, high: int, timeline_script: str) -> list[dict]:
    """Return messages list with the Timeline Script marked for 1h cache.

    Anthropic prompt-caching API expects content blocks. We split the prompt
    into:
      [persona description + role framing]  (NOT cached — varies per persona)
      [Timeline Script]                      (cached — shared across 8 personas)
      [final instruction tail]               (NOT cached — varies per persona)

    The order matches PROMPT.format(...) exactly, just split into 3 blocks
    so we can attach cache_control to the middle one.
    """
    persona_block = (
        "You are a YouTube viewer with this persona:\n"
        f"{persona_desc}\n\n"
        f"Expected comment frequency on a typical 2-minute video: {low}-{high} comments.\n\n"
        "You just finished watching a short video. Below is the complete timeline of what happened:\n\n"
    )
    timeline_block = timeline_script
    tail_block = (
        "\n\nReflect on the entire video. List the moments where you would have left a comment, "
        "staying in character with the persona description above. Match your persona's expected "
        "comment count range above (scaled to the actual video duration).\n\n"
        "Output ONLY a valid JSON array in this exact format:\n"
        "[\n"
        "  {\"timestamp\": \"MM:SS\", \"comment\": \"your comment here\"},\n"
        "  ...\n"
        "]\n\n"
        "If nothing struck you, output an empty array: [].\n\n"
        "Important:\n"
        "- Choose timestamps that fall within actual segment ranges shown above\n"
        "- Each comment should be 10-50 words\n"
        "- This is post-hoc reflection (you finished watching)"
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": persona_block},
                {
                    "type": "text",
                    "text": timeline_block,
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                },
                {"type": "text", "text": tail_block},
            ],
        }
    ]


# ---------- Batch construction ----------

def custom_id_for(video_id: str, persona_id: str) -> str:
    """Deterministic, unique, 1-64 chars per Anthropic spec."""
    return f"{video_id}__{persona_id}"


def parse_custom_id(cid: str) -> tuple[str, str]:
    video_id, persona_id = cid.rsplit("__", 1)
    return video_id, persona_id


def build_batch_requests(
    videos: list[dict],
    personas: list[dict],
    timelines_dir: Path,
    model: str,
    max_tokens: int = 1024,
    skip_done: set[tuple[str, str]] | None = None,
) -> list[dict]:
    """Return Anthropic batch request payload list, ordered by video so cache hits stay warm."""
    skip_done = skip_done or set()
    out: list[dict] = []
    for v in videos:
        ts_path = timelines_dir / f"{v['id']}.txt"
        if not ts_path.exists():
            print(f"  SKIP {v['id']}: missing timeline")
            continue
        timeline = ts_path.read_text()
        for p in personas:
            key = (v["id"], p["id"])
            if key in skip_done:
                continue
            lo, hi = p["expected_comment_count_range"]
            messages = build_messages_with_cache(p["description"], lo, hi, timeline)
            out.append({
                "custom_id": custom_id_for(v["id"], p["id"]),
                "params": {
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": messages,
                },
            })
    return out


# ---------- Batch submission + polling ----------

def submit_batch(client, requests: list[dict]) -> str:
    """POST /v1/messages/batches → batch_id."""
    batch = client.messages.batches.create(requests=requests)
    return batch.id


def poll_batch(client, batch_id: str, poll_interval: int = 30) -> dict:
    """Poll until processing_status='ended'. Returns final batch object."""
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        elapsed = "?"
        if hasattr(batch, "created_at") and batch.created_at:
            try:
                from datetime import datetime, timezone
                created = batch.created_at
                if isinstance(created, str):
                    created = datetime.fromisoformat(created.replace("Z", "+00:00"))
                elapsed = f"{(datetime.now(timezone.utc) - created).total_seconds() / 60:.1f}min"
            except Exception:
                pass
        print(f"  [{batch.processing_status}] elapsed={elapsed} "
              f"processing={counts.processing} succeeded={counts.succeeded} "
              f"errored={counts.errored} canceled={counts.canceled} expired={counts.expired}")
        if batch.processing_status == "ended":
            return batch
        time.sleep(poll_interval)


def stream_results(client, batch_id: str):
    """Generator yielding per-request results from streaming endpoint."""
    for result in client.messages.batches.results(batch_id):
        yield result


# ---------- Main entry ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--personas", required=True)
    ap.add_argument("--timelines-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = all videos; otherwise batch only first N")
    ap.add_argument("--model", default="claude-sonnet-4-5-20250929")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--poll-interval", type=int, default=30)
    ap.add_argument("--no-poll", action="store_true",
                    help="submit batch then exit; resume later with --resume BATCH_ID")
    ap.add_argument("--resume", type=str, default=None,
                    help="poll an existing batch ID instead of submitting a new one")
    args = ap.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        # Fall back to .env in cwd if python-dotenv is available
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(2)

    from anthropic import Anthropic
    client = Anthropic()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Track already-done pairs so we can resume across runs
    done: set[tuple[str, str]] = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add((rec["video_id"], rec["persona_id"]))
                except json.JSONDecodeError:
                    pass
        print(f"resuming with {len(done)} pairs already in output")

    if args.resume:
        batch_id = args.resume
        print(f"[resume] polling existing batch {batch_id}")
    else:
        with Path(args.manifest).open() as f:
            videos = [json.loads(l) for l in f]
        if args.limit:
            videos = videos[: args.limit]
        personas = parse_yaml_personas(Path(args.personas).read_text())
        print(f"Building batch: {len(videos)} videos × {len(personas)} personas")
        requests = build_batch_requests(
            videos, personas, Path(args.timelines_dir),
            model=args.model, max_tokens=args.max_tokens, skip_done=done,
        )
        if not requests:
            print("Nothing to submit (all pairs done).")
            return
        print(f"Submitting batch with {len(requests)} requests "
              f"(payload ≈ {sum(len(json.dumps(r)) for r in requests) / 1024 / 1024:.1f} MB)")
        batch_id = submit_batch(client, requests)
        print(f"Batch submitted: {batch_id}")
        print(f"  Save this ID — to resume polling later, use --resume {batch_id}")

    if args.no_poll:
        print("--no-poll set; exiting. Resume polling later with --resume", batch_id)
        return

    print(f"Polling batch {batch_id} every {args.poll_interval}s...")
    poll_batch(client, batch_id, poll_interval=args.poll_interval)

    print(f"Streaming results from {batch_id}...")
    n_ok = n_scr_fail = n_tvr_fail = n_errored = n_expired = 0
    durations: dict[str, float] = {}
    # Build duration lookup from manifest
    with Path(args.manifest).open() as f:
        for line in f:
            v = json.loads(line)
            durations[v["id"]] = float(v.get("duration_sec") or 120)

    with out_path.open("a") as out_f:
        for result in stream_results(client, batch_id):
            cid = result.custom_id
            video_id, persona_id = parse_custom_id(cid)
            duration = durations.get(video_id, 120.0)

            r_type = result.result.type
            if r_type == "errored":
                err = result.result.error
                rec = {
                    "video_id": video_id, "persona_id": persona_id, "duration_sec": duration,
                    "raw_output": None, "scr_ok": False, "scr_reason": f"batch_errored: {err}",
                    "tvr_ok": False, "tvr_reason": "batch_errored", "parsed": None,
                    "batch_id": batch_id,
                }
                out_f.write(json.dumps(rec) + "\n")
                n_errored += 1
                print(f"  ERROR {cid}: {err}")
                continue
            if r_type == "expired":
                rec = {
                    "video_id": video_id, "persona_id": persona_id, "duration_sec": duration,
                    "raw_output": None, "scr_ok": False, "scr_reason": "batch_expired",
                    "tvr_ok": False, "tvr_reason": "batch_expired", "parsed": None,
                    "batch_id": batch_id,
                }
                out_f.write(json.dumps(rec) + "\n")
                n_expired += 1
                print(f"  EXPIRED {cid}")
                continue
            if r_type != "succeeded":
                print(f"  WARN {cid}: unknown result type {r_type}")
                continue

            msg = result.result.message
            raw = msg.content[0].text if msg.content else ""
            ok, parsed, scr_reason = schema_compliant(raw)
            tvr_ok, tvr_reason = (False, "scr_failed")
            if ok:
                tvr_ok, tvr_reason = timestamp_valid(parsed, duration)
            rec = {
                "video_id": video_id, "persona_id": persona_id, "duration_sec": duration,
                "raw_output": raw, "scr_ok": ok, "scr_reason": scr_reason,
                "tvr_ok": tvr_ok, "tvr_reason": tvr_reason,
                "parsed": parsed if ok else None,
                "batch_id": batch_id,
                "usage": {
                    "input_tokens": getattr(msg.usage, "input_tokens", None),
                    "output_tokens": getattr(msg.usage, "output_tokens", None),
                    "cache_creation_input_tokens": getattr(msg.usage, "cache_creation_input_tokens", None),
                    "cache_read_input_tokens": getattr(msg.usage, "cache_read_input_tokens", None),
                },
            }
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            if ok and tvr_ok:
                n_ok += 1
            elif not ok:
                n_scr_fail += 1
            else:
                n_tvr_fail += 1
            cache_read = rec["usage"]["cache_read_input_tokens"] or 0
            cache_write = rec["usage"]["cache_creation_input_tokens"] or 0
            n = len(parsed) if parsed else 0
            print(f"  {cid}: SCR={ok} TVR={tvr_ok} n={n} cache=W{cache_write}/R{cache_read}")

    total = n_ok + n_scr_fail + n_tvr_fail + n_errored + n_expired
    if total:
        print(f"\nBatch summary ({batch_id}):")
        print(f"  total: {total}")
        print(f"  fully ok (SCR+TVR): {n_ok} ({n_ok/total*100:.1f}%)")
        print(f"  SCR fail: {n_scr_fail}")
        print(f"  TVR fail: {n_tvr_fail}")
        print(f"  errored: {n_errored}")
        print(f"  expired: {n_expired}")
        if n_errored + n_expired > 0:
            print(f"\nResubmit errored/expired with sync distill_claude.py "
                  f"(it will only redo missing pairs).")


if __name__ == "__main__":
    main()
