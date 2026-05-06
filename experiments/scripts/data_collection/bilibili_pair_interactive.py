"""Interactive YT→Bilibili pairing tool for v4.2.2 Anchor B revival.

Strategy:
  - For each YT video in --candidate-list, search Bilibili with multiple
    candidate queries derived from the YT title.
  - Filter Bilibili results by duration (yt_dur ± tolerance).
  - Show user up to 5 candidates per video; user picks 1 or skips.
  - Append the chosen pair to --out-file (jsonl).

Each YT video typically takes 30 sec to 2 min of human time.

Usage:
  python scripts/data_collection/bilibili_pair_interactive.py \
    --candidate-list <file with YT IDs, one per line> \
    --manifest data/raw_videos/manifest.test.jsonl \
    --out data/anchor_b/yt_to_bili.jsonl

Or specify individual YT IDs as args:
  python scripts/data_collection/bilibili_pair_interactive.py \
    --yt-ids 3s7opUfsin8 dKKPXMKnGR4 YF1nuFxGkGk \
    --manifest data/raw_videos/manifest.test.jsonl \
    --out data/anchor_b/yt_to_bili.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.env import load_env

load_env()

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)


def parse_bili_duration(s: str) -> int:
    parts = [int(x) for x in s.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0


def search_bilibili(keyword: str, max_results: int = 20) -> list[dict]:
    params = {
        "search_type": "video",
        "keyword": keyword,
        "order": "totalrank",
        "page": 1,
    }
    url = "https://api.bilibili.com/x/web-interface/search/type?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Cookie": "buvid3=",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"  Bilibili API error: {e}", file=sys.stderr)
        return []
    if data.get("code") != 0:
        return []
    return (data.get("data", {}).get("result", []) or [])[:max_results]


def html_strip(text: str) -> str:
    return re.sub(r"</?em[^>]*>", "", text)


def derive_queries(yt_title: str) -> list[str]:
    """Heuristic: try multiple queries derived from the YT title.

    For food/cooking: extract main ingredient nouns
    For tech: extract product brand + model
    For education: extract topic keyword
    For comedy: extract comedian name
    """
    queries = []
    lower = yt_title.lower()

    # Food keywords
    food_words = re.findall(
        r"\b(tomato|egg|banana|oats?|nacho|cheese|recipe|breakfast|pizza|"
        r"omelet|sauce|cooking|food|meal)\b",
        lower
    )
    if food_words:
        unique_food = list(dict.fromkeys(food_words))[:3]
        queries.append(" ".join(unique_food) + " 食谱")
        queries.append(" ".join(unique_food) + " 早餐")

    # Tech / product
    tech_brands = re.findall(
        r"\b(bose|sony|jbl|apple|samsung|huawei|xiaomi|airpods|earbuds|iphone|"
        r"freebuds|quietcomfort|ultra)\b",
        lower
    )
    if tech_brands:
        queries.append(" ".join(dict.fromkeys(tech_brands))[:50] + " 评测")
        queries.append(" ".join(dict.fromkeys(tech_brands))[:50])

    # Science / education
    sci_words = re.findall(
        r"\b(plate tectonics|tectonics|universe|aurora|northern lights|"
        r"buoyancy|heat transfer|fat loss|burn fat|big bang)\b",
        lower
    )
    if sci_words:
        for w in sci_words[:2]:
            queries.append(w)
        queries.append("科普 " + sci_words[0])

    # Comedy / standup
    comedy_words = re.findall(
        r"\b(stand[- ]?up|comedy|comicstaan|brian regan|zakir khan|aakash gupta|"
        r"lachlan patterson|mel mcdaniel)\b",
        lower
    )
    if comedy_words:
        for w in comedy_words[:2]:
            queries.append(w + " 脱口秀")
            queries.append(w)

    # Generic fallback: first 3 keywords from title
    title_words = re.findall(r"[A-Za-z]+", yt_title)
    if title_words:
        queries.append(" ".join(title_words[:3]))

    # Deduplicate while preserving order
    seen = set()
    out = []
    for q in queries:
        if q.strip() and q not in seen:
            seen.add(q)
            out.append(q.strip())
    return out[:5]  # max 5 queries per video


def collect_candidates(
    yt_video: dict,
    duration_tol: int = 30,
    sleep: float = 0.5,
) -> list[dict]:
    """Search Bilibili with multiple queries, return deduped candidates within
    duration tolerance."""
    queries = derive_queries(yt_video["title"])
    yt_dur = yt_video["duration_sec"]
    seen_bvids: set[str] = set()
    candidates: list[dict] = []

    for q in queries:
        results = search_bilibili(q, max_results=15)
        for r in results:
            bvid = r.get("bvid")
            if not bvid or bvid in seen_bvids:
                continue
            try:
                dur = parse_bili_duration(r.get("duration", "0:00"))
            except Exception:
                continue
            if dur == 0 or abs(dur - yt_dur) > duration_tol:
                continue
            seen_bvids.add(bvid)
            candidates.append({
                "bvid": bvid,
                "title": html_strip(r.get("title", "")),
                "duration_sec": dur,
                "play": r.get("play", 0),
                "pubdate": r.get("pubdate"),
                "description": html_strip(r.get("description", ""))[:200],
                "_query": q,
            })
        time.sleep(sleep)
        if len(candidates) >= 10:
            break

    # Sort by play count desc
    candidates.sort(key=lambda c: -c.get("play", 0))
    return candidates[:8]  # show max 8


def render_candidate(idx: int, cand: dict) -> str:
    return (
        f"  [{idx}] BVID: {cand['bvid']}  "
        f"({cand['duration_sec']}s, {cand['play']:>10,} plays, "
        f"via query={cand['_query']!r})\n"
        f"      Title: {cand['title']}\n"
        f"      Desc:  {cand['description'][:120]}\n"
        f"      URL:   https://www.bilibili.com/video/{cand['bvid']}"
    )


def already_done(out_path: Path) -> set[str]:
    done: set[str] = set()
    if not out_path.exists():
        return done
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                done.add(rec["yt_id"])
            except json.JSONDecodeError:
                continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="manifest.test.jsonl with YT video metadata")
    ap.add_argument("--candidate-list", default=None,
                    help="optional: file with YT IDs (one per line) to pair")
    ap.add_argument("--yt-ids", nargs="*", default=None,
                    help="optional: YT IDs as command-line args")
    ap.add_argument("--out", required=True,
                    help="append-mode jsonl: yt→bili pairs")
    ap.add_argument("--duration-tolerance", type=int, default=30)
    args = ap.parse_args()

    # Load YT manifest
    with Path(args.manifest).open() as f:
        manifest = {json.loads(l)["id"]: json.loads(l) for l in f}

    # Determine which YT IDs to pair
    if args.candidate_list:
        with Path(args.candidate_list).open() as f:
            yt_ids = [line.strip() for line in f if line.strip()]
    elif args.yt_ids:
        yt_ids = args.yt_ids
    else:
        yt_ids = list(manifest.keys())

    # Resume
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = already_done(out_path)
    yt_ids = [vid for vid in yt_ids if vid not in done]
    print(f"Pairing {len(yt_ids)} YT videos to Bilibili (already done: {len(done)})")
    print(f"Output: {out_path}\n")

    n_paired = 0
    n_skipped = 0

    out_f = out_path.open("a", encoding="utf-8")
    try:
        for i, yt_id in enumerate(yt_ids, 1):
            v = manifest.get(yt_id)
            if v is None:
                print(f"[{i}/{len(yt_ids)}] {yt_id}: NOT IN MANIFEST, skipping")
                continue

            print("\n" + "=" * 90)
            print(f"=== Pairing {i}/{len(yt_ids)} ===")
            print(f"YT ID:    {yt_id}")
            print(f"Title:    {v['title']}")
            print(f"Category: {v['category']}")
            print(f"Duration: {v['duration_sec']}s   Views: {v['views']:,}")
            print(f"URL:      https://youtube.com/watch?v={yt_id}")
            print("=" * 90)
            print(f"Searching Bilibili...")

            candidates = collect_candidates(v, duration_tol=args.duration_tolerance)

            if not candidates:
                print("\nNo Bilibili candidates within duration tolerance.")
                rec = {"yt_id": yt_id, "yt_title": v["title"],
                       "bvid": None, "_note": "no candidates"}
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                n_skipped += 1
                continue

            print(f"\nFound {len(candidates)} candidates:\n")
            for idx, cand in enumerate(candidates, 1):
                print(render_candidate(idx, cand))
                print()

            print(f"Open the YT URL above and the Bilibili URL of each candidate")
            print(f"to compare visually. Pick by number, or:")
            print(f"  [n] none match   [s] skip (decide later)   [q] quit")

            while True:
                choice = input("Your choice: ").strip().lower()
                if choice in ("q", "quit"):
                    print(f"\nFinal: paired={n_paired}, skipped={n_skipped}, "
                          f"unpaired (no match)={i - n_paired - n_skipped - 1}")
                    return
                if choice in ("s", "skip"):
                    n_skipped += 1
                    print("  (skipped — file untouched, can re-run later)")
                    break
                if choice in ("n", "none"):
                    rec = {"yt_id": yt_id, "yt_title": v["title"],
                           "bvid": None, "_note": "user marked no match"}
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_f.flush()
                    print("  (recorded as no match)")
                    break
                if choice.isdigit():
                    idx = int(choice)
                    if 1 <= idx <= len(candidates):
                        cand = candidates[idx - 1]
                        rec = {
                            "yt_id": yt_id,
                            "yt_title": v["title"],
                            "yt_duration_sec": v["duration_sec"],
                            "yt_views": v["views"],
                            "bvid": cand["bvid"],
                            "bili_title": cand["title"],
                            "bili_duration_sec": cand["duration_sec"],
                            "bili_play": cand["play"],
                            "bili_pubdate": cand["pubdate"],
                            "search_query_used": cand["_query"],
                        }
                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        out_f.flush()
                        n_paired += 1
                        print(f"  PAIRED: yt={yt_id} ↔ bv={cand['bvid']}")
                        break
                    print(f"  invalid number (must be 1-{len(candidates)})")
                    continue
                print("  unknown choice; try again")
    finally:
        out_f.close()

    print(f"\nFinal: paired={n_paired}, skipped={n_skipped}")
    print(f"\nNext step:")
    print(f"  python scripts/data_collection/collect_bilibili_danmaku.py \\")
    print(f"    --pairs {out_path} \\")
    print(f"    --out-dir data/anchor_b/danmaku_timestamps")


if __name__ == "__main__":
    main()
