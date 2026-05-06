"""Stage A step (b): LLaVA-NeXT 13B segment captions, every 10 seconds.

Spec: SimLens_Research_Plan_v4.1.md §3.2 Step 1.2 (b).
Per 10s segment: extract 4 frames at 0/33/66/100% within the segment, tile into
a 2x2 panel, prompt LLaVA with the panel + that segment's Whisper transcript.

Usage:
  python scripts/stage_a/run_llava.py \
    --manifest data/raw_videos/manifest.en.jsonl \
    --videos-dir data/raw_videos \
    --whisper-dir data/whisper \
    --out-dir data/llava

Output: data/llava/{video_id}.json
  {"video_id": ..., "segments": [{"start": 0, "end": 10, "caption": "..."}, ...]}
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path


def _extract_one_frame(video_path: Path, t: float, out_path: Path) -> bool:
    """Extract one frame, retrying with earlier timestamps if seek fails near end."""
    for attempt_t in (t, max(0.0, t - 0.5), max(0.0, t - 1.0), max(0.0, t - 2.0)):
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-ss", f"{attempt_t:.3f}", "-i", str(video_path),
                    "-frames:v", "1", "-vf", "scale=320:-1", str(out_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError:
            continue
        if out_path.exists() and out_path.stat().st_size > 0:
            return True
    return False


def extract_panel(video_path: Path, start: float, end: float, out_path: Path) -> Path | None:
    """Extract 4 frames at [start..end] 0/33/66/100%, tile 2x2, save as PNG.

    Returns out_path on success, None if any frame can't be extracted (caller
    should skip this segment rather than crash).
    """
    duration = end - start
    times = [start + duration * f for f in (0.0, 0.33, 0.66, 0.99)]
    frames: list[Path] = []
    for i, t in enumerate(times):
        f_out = out_path.with_name(f"{out_path.stem}_f{i}.png")
        if not _extract_one_frame(video_path, t, f_out):
            for f in frames:
                f.unlink(missing_ok=True)
            return None
        frames.append(f_out)

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(frames[0]), "-i", str(frames[1]),
                "-i", str(frames[2]), "-i", str(frames[3]),
                "-filter_complex",
                "[0:v][1:v]hstack=inputs=2[t];[2:v][3:v]hstack=inputs=2[b];[t][b]vstack=inputs=2[v]",
                "-map", "[v]", str(out_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError:
        for f in frames:
            f.unlink(missing_ok=True)
        return None
    for f in frames:
        f.unlink(missing_ok=True)
    return out_path


PROMPT_TEMPLATE = (
    "USER: <image>\nYou are watching a 10-second clip from a YouTube video, shown as a 2x2 panel "
    "of frames in chronological order (top-left → top-right → bottom-left → bottom-right).\n\n"
    "Audio transcript for this clip: \"{transcript}\"\n\n"
    "Describe what is happening in this clip in 80-150 words: setting, subjects, actions, "
    "tone, and any notable visual events. Focus on what a viewer would actually see.\nASSISTANT:"
)


def caption_with_llava(panel_path: Path, transcript_chunk: str, model, processor) -> str:
    from PIL import Image  # type: ignore

    image = Image.open(panel_path).convert("RGB")
    prompt = PROMPT_TEMPLATE.format(transcript=transcript_chunk or "(no audio)")
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    out_ids = model.generate(**inputs, max_new_tokens=350, do_sample=False)
    decoded = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    # LLaVA echoes the prompt; chop everything before the ASSISTANT marker
    if "ASSISTANT:" in decoded:
        return decoded.split("ASSISTANT:", 1)[1].strip()
    return decoded.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--whisper-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ext", default="mp4")
    ap.add_argument("--model-id", default="llava-hf/llava-v1.6-vicuna-13b-hf")
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = all videos; otherwise process first N (sanity check)")
    args = ap.parse_args()

    from transformers import (  # type: ignore
        BitsAndBytesConfig,
        LlavaNextForConditionalGeneration,
        LlavaNextProcessor,
    )
    import torch  # type: ignore

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = out_dir / "_panels"
    panels_dir.mkdir(exist_ok=True)

    print(f"Loading {args.model_id} (4-bit nf4 via bitsandbytes)...", flush=True)
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    processor = LlavaNextProcessor.from_pretrained(args.model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        quantization_config=quant_cfg,
        device_map="auto",
    )
    model.eval()
    # Suppress noisy "max_length and max_new_tokens both set" warning
    if hasattr(model.generation_config, "max_length"):
        model.generation_config.max_length = None
    print(f"  loaded; VRAM used: {torch.cuda.memory_allocated() / 1024**3:.1f} GB",
          flush=True)

    with Path(args.manifest).open() as f:
        videos = [json.loads(l) for l in f]
    if args.limit:
        videos = videos[: args.limit]
    print(f"Processing {len(videos)} videos", flush=True)

    n_done = n_skip = n_missing = 0
    t_overall = time.time()
    for v_idx, v in enumerate(videos, 1):
        out_path = out_dir / f"{v['id']}.json"
        if out_path.exists():
            n_skip += 1
            continue
        whisper_path = Path(args.whisper_dir) / f"{v['id']}.json"
        if not whisper_path.exists():
            print(f"  [{v_idx}/{len(videos)}] WARN: missing whisper for {v['id']}",
                  flush=True)
            n_missing += 1
            continue
        whisper = json.loads(whisper_path.read_text(encoding="utf-8"))
        duration = whisper["duration"]
        n_segments = math.ceil(duration / 10.0)
        video_path = Path(args.videos_dir) / f"{v['id']}.{args.ext}"
        if not video_path.exists():
            print(f"  [{v_idx}/{len(videos)}] WARN: missing mp4 for {v['id']}",
                  flush=True)
            n_missing += 1
            continue

        t_start = time.time()
        captions: list[dict] = []
        n_skipped = 0
        for i in range(n_segments):
            start = i * 10.0
            end = min((i + 1) * 10.0, duration)
            transcript_chunk = " ".join(
                s["text"].strip() for s in whisper["segments"]
                if s["end"] >= start and s["start"] < end
            ).strip()
            panel = panels_dir / f"{v['id']}_seg{i:03d}.png"
            if extract_panel(video_path, start, end, panel) is None:
                captions.append({"start": start, "end": end,
                                 "caption": None,
                                 "error": "ffmpeg_extract_failed"})
                n_skipped += 1
                continue
            try:
                cap = caption_with_llava(panel, transcript_chunk, model, processor)
            except Exception as e:
                cap = None
                captions.append({"start": start, "end": end,
                                 "caption": None,
                                 "error": f"llava_failed: {type(e).__name__}: {e}"})
                n_skipped += 1
                panel.unlink(missing_ok=True)
                continue
            captions.append({"start": start, "end": end, "caption": cap})
            panel.unlink(missing_ok=True)

        out_path.write_text(
            json.dumps({"video_id": v["id"], "segments": captions},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        elapsed = time.time() - t_start
        elapsed_total = time.time() - t_overall
        eta = elapsed_total / v_idx * (len(videos) - v_idx)
        skip_note = f" ({n_skipped} skipped)" if n_skipped else ""
        print(f"  [{v_idx}/{len(videos)}] {v['id']}: {n_segments} segs in {elapsed:.1f}s"
              f"{skip_note}  (ETA {eta/60:.0f} min)", flush=True)
        n_done += 1

    print(f"\nSummary: done={n_done} skipped={n_skip} missing={n_missing}", flush=True)


if __name__ == "__main__":
    main()
