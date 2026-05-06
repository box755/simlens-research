"""Sparse JSON schema for SimLens Stage B output.

Used by:
- Phase 1 SFT: Outlines/XGrammar constrained decoding
- Phase 2 DPO: candidate generation
- Group 0 evaluation: SCR / TVR
"""

from __future__ import annotations

import json
import re
from typing import Any

SPARSE_LIST_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "timestamp": {"type": "string", "pattern": r"^\d{2}:\d{2}$"},
            "comment": {"type": "string", "minLength": 5, "maxLength": 300},
        },
        "required": ["timestamp", "comment"],
        "additionalProperties": False,
    },
}

_TS_RE = re.compile(r"^(\d{2}):(\d{2})$")


def parse_timestamp(ts: str) -> int:
    """MM:SS -> total seconds. Raises ValueError on malformed input."""
    m = _TS_RE.match(ts)
    if not m:
        raise ValueError(f"bad timestamp format: {ts!r}")
    mm, ss = int(m.group(1)), int(m.group(2))
    if ss >= 60:
        raise ValueError(f"seconds >= 60: {ts!r}")
    return mm * 60 + ss


def _strip_json_wrappers(raw: str) -> str:
    """Strip common LLM wrappers around a JSON array:
    - Markdown ```json … ``` fences
    - Leading / trailing prose around a [...] block

    If a [...] substring exists, prefer that. Else return raw stripped.
    """
    s = raw.strip()
    # Triple-backtick fence (with or without "json" tag)
    if s.startswith("```"):
        # remove leading fence + optional language tag
        first_newline = s.find("\n")
        if first_newline != -1:
            s = s[first_newline + 1:]
        # remove trailing fence
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
        s = s.strip()
    # Bracket-extraction fallback: take first [...] block
    first = s.find("[")
    last = s.rfind("]")
    if first != -1 and last > first:
        s = s[first:last + 1]
    return s.strip()


def schema_compliant(raw: str) -> tuple[bool, list[dict] | None, str]:
    """Check if raw model output parses + matches sparse-list schema.

    Tolerates Markdown ```json fences and surrounding prose so long as a
    valid JSON array is recoverable. The training pipeline still uses
    constrained decoding (Outlines/XGrammar) for clean output; this helper
    is robust for evaluating zero-shot baselines whose models often wrap
    JSON in code fences.

    Returns (is_compliant, parsed_list_or_None, reason).
    """
    candidate = _strip_json_wrappers(raw)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as e:
        return False, None, f"json_decode_error: {e}"

    if not isinstance(data, list):
        return False, None, "top_level_not_list"

    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            return False, None, f"entry_{i}_not_dict"
        if set(entry.keys()) != {"timestamp", "comment"}:
            return False, None, f"entry_{i}_keys={sorted(entry.keys())}"
        if not isinstance(entry["timestamp"], str) or not _TS_RE.match(entry["timestamp"]):
            return False, None, f"entry_{i}_bad_timestamp"
        c = entry["comment"]
        if not isinstance(c, str) or not (5 <= len(c) <= 300):
            return False, None, f"entry_{i}_bad_comment_length={len(c) if isinstance(c,str) else 'NA'}"

    return True, data, "ok"


def timestamp_valid(parsed: list[dict], video_duration_sec: float) -> tuple[bool, str]:
    """For SCR-passing output, verify timestamps are in-range, unique, sorted."""
    if not parsed:
        return True, "empty_list_ok"

    seconds = []
    for i, entry in enumerate(parsed):
        try:
            sec = parse_timestamp(entry["timestamp"])
        except ValueError as e:
            return False, f"entry_{i}_parse_error: {e}"
        if sec > video_duration_sec:
            return False, f"entry_{i}_out_of_range: {sec}>{video_duration_sec}"
        seconds.append(sec)

    if len(set(seconds)) != len(seconds):
        return False, "duplicate_timestamps"

    if seconds != sorted(seconds):
        return False, "not_sorted"

    return True, "ok"
