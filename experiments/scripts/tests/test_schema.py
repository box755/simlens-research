"""Unit tests for sparse JSON schema validation.

Run from experiments/:  python -m scripts.tests.test_schema
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.schema import parse_timestamp, schema_compliant, timestamp_valid


def test_parse_timestamp():
    assert parse_timestamp("00:00") == 0
    assert parse_timestamp("01:30") == 90
    assert parse_timestamp("02:59") == 179
    for bad in ["1:30", "01:60", "0130", "ab:cd", ""]:
        try:
            parse_timestamp(bad)
        except ValueError:
            continue
        raise AssertionError(f"should have rejected {bad!r}")


def test_schema_compliant_ok():
    raw = '[{"timestamp":"00:15","comment":"this is a fine comment"}]'
    ok, parsed, reason = schema_compliant(raw)
    assert ok, reason
    assert parsed == [{"timestamp": "00:15", "comment": "this is a fine comment"}]


def test_schema_compliant_empty_list():
    ok, parsed, reason = schema_compliant("[]")
    assert ok and parsed == [], reason


def test_schema_compliant_failures():
    cases = [
        ("not json", "json_decode_error"),
        ('{"timestamp":"00:00","comment":"hi hi"}', "top_level_not_list"),
        ('["foo"]', "entry_0_not_dict"),
        ('[{"timestamp":"00:00"}]', "entry_0_keys"),
        ('[{"timestamp":"0:00","comment":"hello"}]', "bad_timestamp"),
        ('[{"timestamp":"00:00","comment":"hi"}]', "bad_comment_length"),
        ('[{"timestamp":"00:00","comment":"' + "x" * 301 + '"}]', "bad_comment_length"),
    ]
    for raw, expected_substr in cases:
        ok, _, reason = schema_compliant(raw)
        assert not ok, f"should have rejected: {raw}"
        assert expected_substr in reason, f"reason {reason} missing {expected_substr}"


def test_timestamp_valid():
    parsed = [{"timestamp": "00:15", "comment": "x"}, {"timestamp": "01:00", "comment": "y"}]
    assert timestamp_valid(parsed, 120)[0]

    parsed_oor = [{"timestamp": "03:00", "comment": "x"}]
    ok, _ = timestamp_valid(parsed_oor, 120)
    assert not ok

    parsed_dup = [{"timestamp": "00:15", "comment": "a"}, {"timestamp": "00:15", "comment": "b"}]
    ok, _ = timestamp_valid(parsed_dup, 120)
    assert not ok

    parsed_unsorted = [{"timestamp": "01:00", "comment": "a"}, {"timestamp": "00:15", "comment": "b"}]
    ok, _ = timestamp_valid(parsed_unsorted, 120)
    assert not ok

    assert timestamp_valid([], 120)[0]


def main():
    tests = [
        test_parse_timestamp,
        test_schema_compliant_ok,
        test_schema_compliant_empty_list,
        test_schema_compliant_failures,
        test_timestamp_valid,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
