"""Unit tests for distribution-level Anchor B metrics (v4.2.1 Option B)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.anchor_metrics import (
    inter_gap_distribution,
    inter_gap_ks,
    mention_density_ratio,
    mention_depth_distribution,
    mention_depth_ks,
)


def test_mention_depth_distribution():
    mentions = {
        "v1": [30.0, 60.0, 90.0],   # depths 0.25, 0.50, 0.75 (dur=120)
        "v2": [50.0],                # depth 0.50 (dur=100)
    }
    durations = {"v1": 120.0, "v2": 100.0}
    depths = mention_depth_distribution(mentions, durations)
    assert sorted(depths) == [0.25, 0.5, 0.5, 0.75]


def test_mention_depth_ks_identical():
    """Two identical distributions → KS should be 0."""
    mentions = {f"v{i}": [30.0 + i, 60.0 + i, 90.0 + i] for i in range(20)}
    durations = {f"v{i}": 120.0 for i in range(20)}
    ks = mention_depth_ks(mentions, mentions, durations)
    assert ks == 0.0


def test_mention_depth_ks_different():
    """Real mentions front-loaded; predictions back-loaded → high KS."""
    real = {f"v{i}": [10.0, 20.0, 30.0] for i in range(20)}  # depth 0.08-0.25
    pred = {f"v{i}": [90.0, 100.0, 110.0] for i in range(20)}  # depth 0.75-0.92
    durations = {f"v{i}": 120.0 for i in range(20)}
    ks = mention_depth_ks(real, pred, durations)
    print(f"  ks(front, back) = {ks:.3f}  (expect ~1.0)")
    assert ks > 0.9


def test_inter_gap_distribution():
    timestamps = {
        "v1": [10.0, 25.0, 40.0],   # gaps 15, 15
        "v2": [50.0],                # no gaps
        "v3": [100.0, 105.0, 120.0], # gaps 5, 15
    }
    gaps = inter_gap_distribution(timestamps)
    assert sorted(gaps) == [5.0, 15.0, 15.0, 15.0]


def test_inter_gap_ks_identical():
    ts = {f"v{i}": [10.0, 30.0, 50.0] for i in range(10)}
    ks = inter_gap_ks(ts, ts)
    assert ks == 0.0


def test_mention_density_ratio():
    real = {"v1": [10.0, 20.0], "v2": [30.0, 40.0, 50.0]}  # avg 2.5/video
    pred = {"v1": [15.0, 25.0, 35.0], "v2": [45.0]}        # avg 2.0/video
    res = mention_density_ratio(real, pred)
    print(f"  density_ratio: real={res['real_avg_per_video']:.2f} "
          f"pred={res['pred_avg_per_video']:.2f} ratio={res['density_ratio']:.3f}")
    assert abs(res["real_avg_per_video"] - 2.5) < 1e-9
    assert abs(res["pred_avg_per_video"] - 2.0) < 1e-9
    assert abs(res["density_ratio"] - 0.8) < 1e-9


def main():
    tests = [
        test_mention_depth_distribution,
        test_mention_depth_ks_identical,
        test_mention_depth_ks_different,
        test_inter_gap_distribution,
        test_inter_gap_ks_identical,
        test_mention_density_ratio,
    ]
    for t in tests:
        print(f"\n=== {t.__name__} ===")
        t()
        print(f"PASS")
    print(f"\n{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
