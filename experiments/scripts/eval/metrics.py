"""Evaluation metrics for SimLens v4.1 Group 0 + Group 3.

Group 0 (deterministic):
  - SCR (Schema Compliance Rate)
  - TVR (Timestamp Validity Rate)
  - Composite FCR = min(SCR, TVR)

Group 3 (reward-independent):
  - TAS (Teacher Alignment Score, F1@5s) — SoccerNet-style temporal F1.
  - Persona Content Distinctiveness — embedding cosine over per-persona corpora.

Group 2 (Persona Cons. / Linguistic / BERTScore / Coherence / Engagingness)
needs PersonaGym + UniEval + BERTScore — left to a separate file once
those libs are installed on the 5090.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.schema import parse_timestamp, schema_compliant, timestamp_valid


def schema_compliance_rate(raw_outputs: Iterable[str]) -> tuple[float, list[str]]:
    """Fraction of outputs that parse + match schema. Returns (rate, failure_reasons)."""
    outputs = list(raw_outputs)
    if not outputs:
        return 0.0, []
    fails: list[str] = []
    n_ok = 0
    for raw in outputs:
        ok, _, reason = schema_compliant(raw)
        if ok:
            n_ok += 1
        else:
            fails.append(reason)
    return n_ok / len(outputs), fails


def timestamp_validity_rate(
    items: Iterable[tuple[str, float]],
) -> tuple[float, list[str]]:
    """For (raw_output, video_duration_sec) pairs that pass SCR, count TVR-passing.

    Items that fail SCR are excluded from the denominator (per v4.1 §5.2 Group 0:
    TVR is conditional on SCR pass). Returns (rate, failure_reasons).
    """
    items = list(items)
    if not items:
        return 0.0, []
    fails: list[str] = []
    n_ok = 0
    n_eligible = 0
    for raw, dur in items:
        ok, parsed, _ = schema_compliant(raw)
        if not ok:
            continue
        n_eligible += 1
        ts_ok, ts_reason = timestamp_valid(parsed, dur)
        if ts_ok:
            n_ok += 1
        else:
            fails.append(ts_reason)
    if n_eligible == 0:
        return 0.0, ["all_outputs_failed_scr"]
    return n_ok / n_eligible, fails


def teacher_alignment_f1(
    student_list: list[dict],
    teacher_list: list[dict],
    tolerance_sec: int = 5,
) -> float:
    """Temporal F1 @ ±tolerance between student and teacher timestamps.

    Greedy matching: each teacher timestamp can match at most one student timestamp
    (and vice versa) within tolerance window. Borrows SoccerNet (CVPR 2018) action
    spotting paradigm.

    F1 = 2 * P * R / (P + R), where P = TP/(TP+FP) over student, R = TP/(TP+FN) over teacher.
    """
    s_secs = sorted(parse_timestamp(e["timestamp"]) for e in student_list)
    t_secs = sorted(parse_timestamp(e["timestamp"]) for e in teacher_list)

    if not s_secs and not t_secs:
        return 1.0  # both empty → trivially aligned
    if not s_secs or not t_secs:
        return 0.0

    matched_t: set[int] = set()
    tp = 0
    for s in s_secs:
        best: int | None = None
        best_diff = tolerance_sec + 1
        for j, t in enumerate(t_secs):
            if j in matched_t:
                continue
            diff = abs(s - t)
            if diff <= tolerance_sec and diff < best_diff:
                best, best_diff = j, diff
        if best is not None:
            matched_t.add(best)
            tp += 1
    fp = len(s_secs) - tp
    fn = len(t_secs) - tp
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def persona_content_distinctiveness(
    persona_corpora: dict[str, list[str]],
    embed_fn,
) -> float:
    """Avg pairwise cosine distance over 8 persona corpora embeddings.

    persona_corpora: {persona_id: [comment, comment, ...]} — one entry per persona.
    embed_fn: text -> list[float]. SimTube IUI 2025 uses OpenAI text-embedding-3-small.

    Returns 0..1; high → personas write distinct content; low → only superficial differences.
    """
    if len(persona_corpora) < 2:
        return 0.0
    persona_ids = sorted(persona_corpora.keys())
    embeds: dict[str, list[float]] = {}
    for pid in persona_ids:
        concat = "\n".join(persona_corpora[pid])
        if not concat.strip():
            embeds[pid] = []
        else:
            embeds[pid] = embed_fn(concat)

    distances: list[float] = []
    for i, pi in enumerate(persona_ids):
        for pj in persona_ids[i + 1 :]:
            ei, ej = embeds[pi], embeds[pj]
            if not ei or not ej:
                continue
            distances.append(1.0 - _cosine_similarity(ei, ej))
    if not distances:
        return 0.0
    return sum(distances) / len(distances)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("embedding dim mismatch")
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
