"""End-to-end dry-run on dummy data: schema → rewards → metrics, all wired together.

Uses fake Claude/Qwen outputs to prove the post-distillation post-DPO eval loop
will work once the real models are connected. No GPU, no API keys needed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.schema import schema_compliant, timestamp_valid
from stage_b_phase2.rewards import reward_total
from eval.metrics import (
    persona_content_distinctiveness,
    schema_compliance_rate,
    teacher_alignment_f1,
    timestamp_validity_rate,
)


# Fake "post-publish" data we'd see from a real Phase 2 run on 1 video × 8 personas.
TIMELINE = """=== Timeline Script for video VID01 (duration 120.0s) ===
[00:00-00:10] Visual: host with a phone, sitting at a desk
              Audio: hey everyone today we're testing this new phone
[00:10-00:20] Visual: hands removing the back cover, a screw falls
              Audio: oh no I dropped a screw
[00:20-00:30] Visual: phone face down on a wooden floor
              Audio: that doesn't sound good
[00:30-00:40] Visual: cracked screen close-up
              Audio: yeah the screen is shattered
[00:40-00:50] Visual: host laughing nervously, holding phone up
              Audio: wow this is the worst review ever
[00:50-01:00] Visual: outro card with subscribe button
              Audio: anyway if you liked this please subscribe
=== End ==="""

VIDEO_DURATION = 60.0  # short clip used for the dry-run

PERSONAS = {
    f"P{i}": {
        "id": f"P{i}",
        "description": f"persona {i} stub description",
        "expected_comment_count_range": [1, 3] if i % 2 else [3, 5],
    }
    for i in range(1, 9)
}

# Pretend Claude (teacher) gave these timestamps:
TEACHER_OUTPUT = [
    {"timestamp": "00:25", "comment": "phone fell that's painful to watch ouch"},
    {"timestamp": "00:35", "comment": "cracked screen omg what a moment for this video"},
]


def fake_student_output(persona_id: str) -> list[dict]:
    """Fake Phase 2 candidate per persona (varying timestamps to exercise eval)."""
    base = int(persona_id[1])
    return [
        {"timestamp": f"00:{20 + base:02d}", "comment": f"P{base}-style reaction at the drop moment"},
        {"timestamp": f"00:{34 + (base % 3):02d}", "comment": f"P{base}-style cracked-screen comment line"},
    ]


def stub_judge(prompt: str) -> int:
    """Mid-scale rating regardless of prompt — fine for wiring test."""
    return 4


def stub_embed(text: str) -> list[float]:
    """Toy hash embedding so distinctiveness gives a non-trivial signal."""
    h = abs(hash(text))
    return [(h >> i) & 0xFF for i in range(0, 64, 8)]


def main():
    raw_outputs: list[str] = []
    parsed_lists: dict[str, list[dict]] = {}

    print("[1] Phase 2 candidate generation (simulated):")
    for pid in PERSONAS:
        sl = fake_student_output(pid)
        raw = '[' + ",".join(
            '{"timestamp":"%s","comment":"%s"}' % (e["timestamp"], e["comment"])
            for e in sl
        ) + ']'
        raw_outputs.append(raw)
        ok, parsed, reason = schema_compliant(raw)
        assert ok, f"{pid} SCR fail: {reason}"
        parsed_lists[pid] = parsed
    print(f"  generated {len(raw_outputs)} sparse lists")

    print("\n[2] Group 0 — Format compliance:")
    scr, _ = schema_compliance_rate(raw_outputs)
    tvr, _ = timestamp_validity_rate([(r, VIDEO_DURATION) for r in raw_outputs])
    fcr = min(scr, tvr)
    print(f"  SCR={scr:.2f}  TVR={tvr:.2f}  FCR={fcr:.2f}")
    assert scr == 1.0, "expected all SCR pass on dummy data"

    print("\n[3] v4.2 2-aspect rewards per persona:")
    all_rewards = {}
    for pid, sl in parsed_lists.items():
        rewards = reward_total(sl, PERSONAS[pid], TIMELINE, VIDEO_DURATION, stub_judge)
        all_rewards[pid] = rewards
        print(f"  {pid}: total={rewards['R_total']:.3f}  "
              f"timing={rewards['R_timing']:.2f}  "
              f"content={rewards['R_content_quality']:.2f}")
    avg_total = sum(r["R_total"] for r in all_rewards.values()) / len(all_rewards)
    print(f"  -> R_total avg = {avg_total:.3f}")

    print("\n[4] Group 3 — TAS (vs fake teacher) per persona:")
    tas_scores = []
    for pid, sl in parsed_lists.items():
        tas = teacher_alignment_f1(sl, TEACHER_OUTPUT, tolerance_sec=5)
        tas_scores.append(tas)
        print(f"  {pid}: TAS@5s = {tas:.2f}")
    print(f"  -> TAS avg = {sum(tas_scores)/len(tas_scores):.2f}")

    print("\n[5] Group 3 — Persona Content Distinctiveness:")
    corpora = {pid: [e["comment"] for e in sl] for pid, sl in parsed_lists.items()}
    dist = persona_content_distinctiveness(corpora, stub_embed)
    print(f"  distinctiveness = {dist:.4f}")
    assert 0.0 <= dist <= 1.0

    print("\n[6] Pipeline preference pair construction (simulated 4 candidates):")
    fake_candidates = {
        "P1": {
            "cand_0": fake_student_output("P1"),
            "cand_1": fake_student_output("P3"),  # different persona's pattern
            "cand_2": fake_student_output("P5"),
            "cand_3": [],  # empty → R_timing empty-list branch + R_content_quality 0.5
        }
    }
    for pid, cands in fake_candidates.items():
        scored = {
            cid: reward_total(sl, PERSONAS[pid], TIMELINE, VIDEO_DURATION, stub_judge)["R_total"]
            for cid, sl in cands.items()
        }
        chosen_id = max(scored, key=scored.get)
        rejected_id = min(scored, key=scored.get)
        print(f"  {pid}: chosen={chosen_id} ({scored[chosen_id]:.3f}) "
              f"rejected={rejected_id} ({scored[rejected_id]:.3f})")
        assert chosen_id != rejected_id

    print("\nAll wiring checks passed.")


if __name__ == "__main__":
    main()
