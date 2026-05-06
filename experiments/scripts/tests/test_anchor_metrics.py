"""Sanity tests for anchor_metrics.py.

Two checks:
  1. Real-vs-real split: split HF 176K corpus in half, compute distances.
     Expectation: all 4 distances ≈ 0 (same distribution).
  2. Real-vs-synthetic toy: a clearly-different "fake SimLens" corpus should
     give visibly larger distances.

Run from experiments/:
  python scripts/tests/test_anchor_metrics.py
"""

import json
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.anchor_metrics import (
    extract_timestamp_mentions,
    kde_peaks,
    length_ks,
    per_video_hotspot_recall_precision,
)


def load_hf_comments(n: int = 10000, seed: int = 42) -> list[str]:
    """Load N random YouTube comments from local HF cache."""
    path = Path("data/anchor_a/hf_yt_comments_180k.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run HF dataset download first")
    rng = random.Random(seed)
    all_texts: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            all_texts.append(json.loads(line)["text"])
    rng.shuffle(all_texts)
    return all_texts[:n]


def test_length_ks_real_vs_real():
    """Random split of real corpus → KS should be ~0."""
    rng = random.Random(0)
    corpus = load_hf_comments(20000)
    rng.shuffle(corpus)
    half = len(corpus) // 2
    a, b = corpus[:half], corpus[half:]
    ks = length_ks(a, b)
    print(f"  length_ks(real, real) = {ks:.4f}  (expect <0.05)")
    assert ks < 0.05, f"real-vs-real KS too high: {ks}"


def test_length_ks_real_vs_synthetic():
    """Synthetic corpus (all 200-char lorem ipsum) should give large KS."""
    real = load_hf_comments(2000)
    synthetic = ["lorem ipsum dolor sit amet, " * 8] * 2000  # ~225 chars each
    ks = length_ks(synthetic, real)
    print(f"  length_ks(synthetic, real) = {ks:.4f}  (expect >0.5)")
    assert ks > 0.5, f"synthetic-vs-real KS too low: {ks}"


def test_extract_timestamp_mentions():
    cases = [
        ("Check out 1:23 it's amazing", 180, [83.0]),
        ("@2:45 omg poor chef", 180, [165.0]),
        ("3:14 PM was when I watched", 180, [194.0]),  # 194 > 180 → filtered
        ("Multiple: 0:30, 1:15 and 2:45", 180, [30.0, 75.0, 165.0]),
        ("No timestamps here lol", 180, []),
    ]
    for text, dur, expected in cases:
        got = extract_timestamp_mentions(text, dur)
        # Filter out any > duration (with tolerance) — note "194" should be filtered
        expected_filtered = [e for e in expected if e <= dur + 5]
        print(f"  extract({text!r}, {dur}) = {got}  expect {expected_filtered}")
        assert got == expected_filtered, f"extract mismatch: got {got}, expect {expected_filtered}"


def test_kde_peaks():
    # 3 clusters of timestamps should yield 3 peaks
    timestamps = [10.0, 11.0, 11.5, 12.0,
                  50.0, 50.5, 51.0,
                  90.0, 91.0]
    peaks = kde_peaks(timestamps, video_duration_sec=120.0, top_k=3)
    print(f"  kde_peaks: {peaks}")
    assert len(peaks) == 3
    # Peaks should be roughly at the cluster centers
    assert any(abs(p - 11) < 3 for p in peaks)
    assert any(abs(p - 50) < 3 for p in peaks)
    assert any(abs(p - 90) < 3 for p in peaks)


def test_hotspot_recall_precision():
    real = [12.0, 50.0, 90.0]
    pred_perfect = [13.0, 50.0, 88.0]  # all within 5s
    pred_overshoot = [13.0, 25.0, 50.0, 60.0, 88.0, 95.0]  # 3 hits + 3 misses
    pred_zero = [25.0, 60.0, 100.0]  # no hits

    r, p = per_video_hotspot_recall_precision(real, pred_perfect)
    print(f"  perfect: recall={r:.2f} prec={p:.2f}")
    assert r == 1.0 and p == 1.0

    r, p = per_video_hotspot_recall_precision(real, pred_overshoot)
    print(f"  overshoot: recall={r:.2f} prec={p:.2f}  (expect r=1, p=0.5)")
    assert r == 1.0 and abs(p - 0.5) < 0.01

    r, p = per_video_hotspot_recall_precision(real, pred_zero)
    print(f"  zero: recall={r:.2f} prec={p:.2f}  (expect 0,0)")
    assert r == 0.0 and p == 0.0


def main():
    tests = [
        test_length_ks_real_vs_real,
        test_length_ks_real_vs_synthetic,
        test_extract_timestamp_mentions,
        test_kde_peaks,
        test_hotspot_recall_precision,
    ]
    for t in tests:
        print(f"\n=== {t.__name__} ===")
        t()
        print(f"PASS")
    print(f"\n{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
