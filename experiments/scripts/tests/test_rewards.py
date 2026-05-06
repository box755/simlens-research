"""Unit tests for v4.2 2-aspect rewards (mock judge). Run from experiments/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage_b_phase2.rewards import (
    reward_content_quality,
    reward_timing,
    reward_total,
)


SAMPLE_TIMELINE = """=== Timeline Script ===
[00:00-00:10] Visual: a host introducing a phone | Audio: hello everyone today
[00:10-00:20] Visual: hands unboxing a phone | Audio: let's see what's inside
[00:20-00:30] Visual: phone falls onto the floor | Audio: oh no
[00:30-00:40] Visual: cracked screen close-up | Audio: this is bad
[00:40-00:50] Visual: host laughing nervously | Audio: I can't believe this
[00:50-01:00] Visual: ending screen | Audio: subscribe for more
=== End ==="""


def stub_judge(prompt: str) -> int:
    """Return a deterministic mid-scale rating regardless of prompt."""
    return 4


def test_reward_timing_with_stub_judge():
    sl = [{"timestamp": "00:25", "comment": "phone fell omg"}]
    score = reward_timing(sl, SAMPLE_TIMELINE, stub_judge)
    assert score == 4 / 5.0


def test_reward_timing_empty_list_no_peaks():
    """Empty list with stub judge counting 4 peaks → 0.5 - 0.1*4 = 0.1."""
    score = reward_timing([], SAMPLE_TIMELINE, stub_judge)
    assert 0.0 <= score <= 1.0


def test_reward_content_quality_with_stub_judge():
    sl = [{"timestamp": "00:25", "comment": "phone fell omg poor host"}]
    persona = {"description": "youthful kpop fan, very emotional"}
    score = reward_content_quality(sl, persona, SAMPLE_TIMELINE, stub_judge)
    assert score == 4 / 5.0


def test_reward_content_quality_empty_list_neutral():
    persona = {"description": "anything"}
    score = reward_content_quality([], persona, SAMPLE_TIMELINE, stub_judge)
    assert score == 0.5


def test_reward_total_v42_weights():
    """v4.2: R_total = 0.50 R_timing + 0.50 R_content_quality"""
    persona = {"description": "extroverted vlog viewer",
               "expected_comment_count_range": [3, 6]}
    sl = [
        {"timestamp": "00:20", "comment": "phone dropped omg this is hilarious"},
        {"timestamp": "00:40", "comment": "the cracked screen reaction is gold"},
    ]
    res = reward_total(sl, persona, SAMPLE_TIMELINE, 60.0, stub_judge)
    expected = 0.50 * res["R_timing"] + 0.50 * res["R_content_quality"]
    assert abs(res["R_total"] - expected) < 1e-9
    # Only 2 aspects + total in result dict
    assert set(res.keys()) == {"R_timing", "R_content_quality", "R_total"}
    for k in ("R_timing", "R_content_quality"):
        assert 0.0 <= res[k] <= 1.0


def test_no_old_v41_rewards_imported():
    """Catch accidental re-introduction of v4.1's R_frequency_match / R_coverage_diversity."""
    from stage_b_phase2 import rewards
    assert not hasattr(rewards, "reward_frequency_match"), \
        "R_frequency_match must NOT exist — removed in v4.2"
    assert not hasattr(rewards, "reward_coverage_diversity"), \
        "R_coverage_diversity must NOT exist — removed in v4.2"


def main():
    tests = [
        test_reward_timing_with_stub_judge,
        test_reward_timing_empty_list_no_peaks,
        test_reward_content_quality_with_stub_judge,
        test_reward_content_quality_empty_list_neutral,
        test_reward_total_v42_weights,
        test_no_old_v41_rewards_imported,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
