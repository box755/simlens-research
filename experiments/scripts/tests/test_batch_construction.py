"""Unit test for batch request construction (no API calls)."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage_b_phase1.distill_claude_batch import (
    build_batch_requests,
    build_messages_with_cache,
    custom_id_for,
    parse_custom_id,
)


def test_custom_id_roundtrip():
    cid = custom_id_for("vid01", "P3")
    assert cid == "vid01__P3"
    v, p = parse_custom_id(cid)
    assert v == "vid01" and p == "P3"


def test_messages_have_cache_control():
    messages = build_messages_with_cache(
        persona_desc="i love cats", low=2, high=4,
        timeline_script="[00:00-00:10] Visual: ... | Audio: ...",
    )
    assert len(messages) == 1
    content = messages[0]["content"]
    assert len(content) == 3
    # Block 0 = persona framing, no cache
    assert "i love cats" in content[0]["text"]
    assert "cache_control" not in content[0]
    # Block 1 = Timeline Script with 1h cache
    assert "[00:00-00:10]" in content[1]["text"]
    assert content[1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    # Block 2 = tail instruction, no cache
    assert "JSON array" in content[2]["text"]
    assert "cache_control" not in content[2]


def test_batch_requests_skip_done_pairs():
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        videos = [{"id": "v1", "duration_sec": 60}, {"id": "v2", "duration_sec": 60}]
        # Write fake timeline files
        for v in videos:
            (td_path / f"{v['id']}.txt").write_text(f"timeline for {v['id']}")
        personas = [
            {"id": "P1", "description": "p1 desc", "expected_comment_count_range": [1, 2]},
            {"id": "P2", "description": "p2 desc", "expected_comment_count_range": [3, 5]},
        ]
        skip = {("v1", "P1")}
        requests = build_batch_requests(
            videos, personas, td_path,
            model="claude-sonnet-4-5-20250929", skip_done=skip,
        )
        # 2 videos × 2 personas = 4, minus 1 skipped = 3
        assert len(requests) == 3
        cids = [r["custom_id"] for r in requests]
        assert "v1__P1" not in cids
        assert "v1__P2" in cids
        assert "v2__P1" in cids
        assert "v2__P2" in cids
        # Each request has the right shape
        for r in requests:
            assert "params" in r
            assert r["params"]["model"] == "claude-sonnet-4-5-20250929"
            assert r["params"]["max_tokens"] == 1024
            assert len(r["params"]["messages"]) == 1


def test_batch_payload_size():
    """Verify a 100-video × 8-persona batch fits well under 256 MB."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        # Realistic Timeline Script (~3000 chars / ~750 tokens)
        big_timeline = "[00:00-00:10] " + ("x" * 3000)
        videos = [{"id": f"v{i:03d}", "duration_sec": 120} for i in range(100)]
        for v in videos:
            (td_path / f"{v['id']}.txt").write_text(big_timeline)
        personas = [
            {"id": f"P{i}", "description": "p" * 200,
             "expected_comment_count_range": [1, 4]} for i in range(1, 9)
        ]
        requests = build_batch_requests(
            videos, personas, td_path, model="claude-sonnet-4-5-20250929",
        )
        assert len(requests) == 800
        size_mb = sum(len(json.dumps(r)) for r in requests) / 1024 / 1024
        assert size_mb < 256, f"Batch too large: {size_mb:.1f} MB"
        print(f"  100×8 batch payload: {size_mb:.1f} MB (limit 256 MB)")


def main():
    tests = [
        test_custom_id_roundtrip,
        test_messages_have_cache_control,
        test_batch_requests_skip_done_pairs,
        test_batch_payload_size,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
