"""2-aspect list-level rewards for SimLens Phase 2 RLAIF DPO.

Spec: SimLens_Research_Plan_v4.2.md §4.3 (was §4.1 in v4.1).

R_total = 0.50 R_timing + 0.50 R_content_quality

v4.1 → v4.2 changelog:
  - REMOVED R_frequency_match: expected_comment_count is Claude-estimated, so
    using it as a reward signal forms a self-referential loop ("Claude-estimated
    activity → trains Student-distilled-from-Claude"). Activity range is now
    a prompt-level constraint only (in §4.3 of v4.2 plan).
  - REMOVED R_coverage_diversity: the "timestamps should spread evenly"
    assumption contradicts the sparse event-driven motivation. Real reaction
    moments cluster around peaks; uniform-spacing reward penalizes correct
    behavior. Spread is now implicitly regulated by R_timing.

Both R_timing and R_content_quality call Qwen3-32B-Q4 via Ollama. Judge calls
are abstracted behind `judge_fn` so tests can pass a mock and the real run can
swap in Ollama.
"""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.schema import parse_timestamp

JudgeFn = Callable[[str], int]  # prompt -> 1..5 integer


def reward_timing(
    sparse_list: list[dict],
    timeline_script: str,
    judge_fn: JudgeFn,
    window_sec: int = 5,
) -> float:
    """R_timing — saliency of each predicted timestamp (SoccerNet-style)."""
    if not sparse_list:
        peaks = _count_high_engagement_segments(timeline_script, judge_fn)
        return 1.0 if peaks == 0 else max(0.0, 0.5 - 0.1 * peaks)

    scores: list[float] = []
    for entry in sparse_list:
        ts_sec = parse_timestamp(entry["timestamp"])
        local = _segment_window(timeline_script, ts_sec, window_sec)
        prompt = (
            f"You are evaluating whether a YouTube viewer would naturally comment at this moment.\n\n"
            f"Local context (±{window_sec}s around {entry['timestamp']}):\n{local}\n\n"
            "On a 1-5 scale, rate whether this moment contains a noteworthy event "
            "(highlight, twist, climax, surprise, emotional peak):\n"
            "1: completely uneventful\n2: minor event\n3: moderate event\n"
            "4: clear highlight\n5: strong climax/twist\n\nOutput ONLY the integer."
        )
        scores.append(int(judge_fn(prompt)) / 5.0)
    return mean(scores)


def reward_content_quality(
    sparse_list: list[dict],
    persona: dict,
    timeline_script: str,
    judge_fn: JudgeFn,
    window_sec: int = 5,
) -> float:
    """R_content_quality — per-comment Persona Cons. + Linguistic + Local Relevance, averaged.

    v4.2: empty list returns 0.5 (neutral). Whether list should be empty is
    decided by R_timing's empty-list branch, not by R_frequency_match (removed).
    """
    if not sparse_list:
        return 0.5

    scores: list[float] = []
    for entry in sparse_list:
        ts_sec = parse_timestamp(entry["timestamp"])
        local = _segment_window(timeline_script, ts_sec, window_sec)
        prompt = (
            f"You are evaluating a YouTube comment on three dimensions.\n\n"
            f"Persona description:\n{persona}\n\n"
            f"Local video context (±{window_sec}s around {entry['timestamp']}):\n{local}\n\n"
            f"Generated comment:\n\"{entry['comment']}\"\n\n"
            "Rate 1-5 on each dimension, then output the AVERAGE as a single integer:\n"
            "A. Persona Consistency (PersonaGym rubric): 1=contradicts persona; 5=strongly reflects.\n"
            "B. Linguistic Habits: 1=tone mismatch; 5=perfectly matches persona style.\n"
            "C. Local Relevance: 1=refers to off-segment events; 5=directly responds to this 5s window.\n\n"
            "Output ONLY the average integer (1-5)."
        )
        scores.append(int(judge_fn(prompt)) / 5.0)
    return mean(scores)


def reward_total(
    sparse_list: list[dict],
    persona: dict,
    timeline_script: str,
    video_duration_sec: float,
    judge_fn: JudgeFn,
) -> dict[str, float]:
    """Compute weighted total + per-aspect breakdown.

    v4.2: 2-aspect with equal 50/50 weights.
    The `persona` and `video_duration_sec` args are kept in the signature for
    backward compatibility with callers that pass them; they are not used by
    the simplified reward_timing / reward_content_quality.
    """
    _ = persona  # unused in v4.2 R_total (was used by removed R_frequency_match)
    _ = video_duration_sec  # unused in v4.2 R_total (was used by removed R_coverage_diversity)
    r_t = reward_timing(sparse_list, timeline_script, judge_fn)
    r_c = reward_content_quality(sparse_list, persona, timeline_script, judge_fn)
    total = 0.50 * r_t + 0.50 * r_c
    return {
        "R_timing": r_t,
        "R_content_quality": r_c,
        "R_total": total,
    }


def _segment_window(timeline_script: str, center_sec: int, window_sec: int) -> str:
    """Extract Timeline Script lines whose [MM:SS-MM:SS] segment overlaps [center-w, center+w].

    Falls back to whole script on parse failure (judge still gets coherent context).
    """
    lo, hi = max(0, center_sec - window_sec), center_sec + window_sec
    out_lines: list[str] = []
    in_segment = False
    seg_overlaps = False
    for line in timeline_script.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and "-" in stripped[:20]:
            seg_overlaps = _segment_header_overlaps(stripped, lo, hi)
            in_segment = seg_overlaps
            if seg_overlaps:
                out_lines.append(line)
        elif in_segment:
            out_lines.append(line)
    return "\n".join(out_lines) if out_lines else timeline_script


def _segment_header_overlaps(header: str, lo: int, hi: int) -> bool:
    """Header form: '[00:00-00:10] ...'. Returns True if [start,end] overlaps [lo,hi]."""
    try:
        bracket = header.split("]", 1)[0].lstrip("[")
        start_str, end_str = bracket.split("-")
        start = parse_timestamp(start_str.strip())
        end = parse_timestamp(end_str.strip())
    except (ValueError, IndexError):
        return False
    return not (end < lo or start > hi)


def _count_high_engagement_segments(timeline_script: str, judge_fn: JudgeFn) -> int:
    """Empty-list fallback: ask judge to count peaks in whole script (cheap heuristic)."""
    prompt = (
        "Count the number of segments in this Timeline Script that contain a highlight, twist, "
        "climax, or strong emotional peak (events a YouTube viewer would naturally comment on).\n\n"
        f"{timeline_script}\n\nOutput ONLY a single integer (0-20)."
    )
    try:
        return max(0, int(judge_fn(prompt)))
    except (ValueError, TypeError):
        return 0
