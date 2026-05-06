"""Unit tests for eval metrics. Run from experiments/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import (
    persona_content_distinctiveness,
    schema_compliance_rate,
    teacher_alignment_f1,
    timestamp_validity_rate,
)


def test_scr_all_pass():
    raw = [
        '[{"timestamp":"00:15","comment":"hello world"}]',
        "[]",
        '[{"timestamp":"00:30","comment":"another comment"}]',
    ]
    rate, fails = schema_compliance_rate(raw)
    assert rate == 1.0 and not fails


def test_scr_mixed():
    raw = [
        '[{"timestamp":"00:15","comment":"hello world"}]',
        "not json",
        '[{"timestamp":"00:30"}]',
    ]
    rate, fails = schema_compliance_rate(raw)
    assert abs(rate - 1 / 3) < 1e-9
    assert len(fails) == 2


def test_tvr_skips_scr_failures():
    items = [
        ('[{"timestamp":"00:15","comment":"hello world"}]', 120),
        ("not json", 120),
        ('[{"timestamp":"05:00","comment":"out of range"}]', 120),
    ]
    rate, _ = timestamp_validity_rate(items)
    # Eligible = 2 (one SCR fail dropped); 1 of 2 passes TVR
    assert rate == 0.5


def test_tas_perfect_match():
    s = [{"timestamp": "00:10", "comment": "x"}, {"timestamp": "00:30", "comment": "y"}]
    t = [{"timestamp": "00:10", "comment": "a"}, {"timestamp": "00:30", "comment": "b"}]
    assert teacher_alignment_f1(s, t) == 1.0


def test_tas_within_tolerance():
    s = [{"timestamp": "00:13", "comment": "x"}]  # within 5s of 00:10
    t = [{"timestamp": "00:10", "comment": "a"}]
    assert teacher_alignment_f1(s, t, tolerance_sec=5) == 1.0


def test_tas_outside_tolerance():
    s = [{"timestamp": "00:20", "comment": "x"}]  # 10s away
    t = [{"timestamp": "00:10", "comment": "a"}]
    assert teacher_alignment_f1(s, t, tolerance_sec=5) == 0.0


def test_tas_partial():
    s = [
        {"timestamp": "00:10", "comment": "x"},
        {"timestamp": "00:30", "comment": "y"},
        {"timestamp": "00:50", "comment": "z"},
    ]
    t = [{"timestamp": "00:10", "comment": "a"}, {"timestamp": "00:30", "comment": "b"}]
    # Student: TP=2, FP=1 → P=2/3
    # Teacher: TP=2, FN=0 → R=1.0
    # F1 = 2 * (2/3) * 1 / (2/3 + 1) = (4/3) / (5/3) = 0.8
    assert abs(teacher_alignment_f1(s, t) - 0.8) < 1e-9


def test_tas_both_empty():
    assert teacher_alignment_f1([], []) == 1.0


def test_tas_one_empty():
    assert teacher_alignment_f1([{"timestamp": "00:10", "comment": "x"}], []) == 0.0


def test_persona_distinctiveness_high_for_orthogonal_embeddings():
    corpora = {"P1": ["a"], "P2": ["b"], "P3": ["c"]}
    # Mock embeds: pairwise orthogonal unit vectors → cosine sim = 0 → distance = 1
    embeds = {"a": [1, 0, 0], "b": [0, 1, 0], "c": [0, 0, 1]}
    score = persona_content_distinctiveness(corpora, lambda txt: embeds[txt])
    assert abs(score - 1.0) < 1e-9


def test_persona_distinctiveness_low_for_identical_embeddings():
    corpora = {"P1": ["x"], "P2": ["y"], "P3": ["z"]}
    embeds = {"x": [1, 0], "y": [1, 0], "z": [1, 0]}
    score = persona_content_distinctiveness(corpora, lambda txt: embeds[txt])
    assert abs(score) < 1e-9


def main():
    tests = [
        test_scr_all_pass,
        test_scr_mixed,
        test_tvr_skips_scr_failures,
        test_tas_perfect_match,
        test_tas_within_tolerance,
        test_tas_outside_tolerance,
        test_tas_partial,
        test_tas_both_empty,
        test_tas_one_empty,
        test_persona_distinctiveness_high_for_orthogonal_embeddings,
        test_persona_distinctiveness_low_for_identical_embeddings,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
