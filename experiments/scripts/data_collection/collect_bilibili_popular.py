"""Collect 50 Bilibili popular 1-3min short videos for v4.2.2 Anchor B-2.

Spec: SimLens_Research_Plan_v4.2.md §3.2 Step 1.4 Anchor B-2.
Academic basis: Refs [38] DanmakuTPPBench (NeurIPS 2025), [39] Plot-Aligned Danmaku.

Strategy:
  - Hit Bilibili popular API (multiple pages) until we have 50 videos with
    duration 60-180s.
  - Output a manifest jsonl with bvid + cid + duration for downstream
    danmaku timestamp extraction.

No correspondence to YouTube videos required — Anchor B-2 is a fully
independent cross-platform temporal anchor (distribution-level KS).

Usage:
  python scripts/data_collection/collect_bilibili_popular.py \
    --out data/anchor_b2/bili_popular_manifest.jsonl \
    --target 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)


def http_get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Cookie": "buvid3=",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_popular_page(page: int, page_size: int = 50) -> list[dict]:
    """GET https://api.bilibili.com/x/web-interface/popular?ps=PS&pn=PAGE."""
    params = {"ps": page_size, "pn": page}
    url = "https://api.bilibili.com/x/web-interface/popular?" + urllib.parse.urlencode(params)
    raw = http_get(url)
    data = json.loads(raw)
    if data.get("code") != 0:
        return []
    return data.get("data", {}).get("list", []) or []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=50,
                    help="how many 60-180s videos to collect")
    ap.add_argument("--max-pages", type=int, default=10,
                    help="upper bound on popular pages to scan")
    ap.add_argument("--page-size", type=int, default=50)
    ap.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_bvids: set[str] = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                seen_bvids.add(json.loads(line)["bvid"])
        print(f"resuming with {len(seen_bvids)} videos already in {out_path}")

    kept: list[dict] = []
    out_f = out_path.open("a", encoding="utf-8")

    try:
        for page in range(1, args.max_pages + 1):
            if len(kept) + len(seen_bvids) >= args.target:
                break
            print(f"[page {page}] fetching popular...")
            try:
                items = fetch_popular_page(page, args.page_size)
            except Exception as e:
                print(f"  page {page} failed: {e}", file=sys.stderr)
                continue
            if not items:
                break

            for item in items:
                bvid = item.get("bvid")
                if not bvid or bvid in seen_bvids:
                    continue
                duration = int(item.get("duration", 0))  # seconds
                if not (60 <= duration <= 180):
                    continue
                rec = {
                    "bvid": bvid,
                    "cid": item.get("cid"),
                    "title": item.get("title", ""),
                    "duration_sec": duration,
                    "view": (item.get("stat") or {}).get("view"),
                    "danmaku": (item.get("stat") or {}).get("danmaku"),
                    "tname": item.get("tname"),
                    "owner_name": (item.get("owner") or {}).get("name"),
                    "pubdate": item.get("pubdate"),
                }
                seen_bvids.add(bvid)
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                kept.append(rec)
                # ASCII-safe print (Windows cp950 can't render Chinese tname)
                print(f"  KEEP {bvid}: {duration}s, "
                      f"{rec['view']:>10,} views, danmaku={rec['danmaku']}")
                if len(kept) + (len(seen_bvids) - len(kept)) >= args.target:
                    break
            time.sleep(args.sleep)
    finally:
        out_f.close()

    total = len(seen_bvids)
    print(f"\nSummary: total kept (cumulative) = {total}, this run added {len(kept)}")
    if total < args.target:
        print(f"  WARN: only {total}/{args.target} target — "
              f"try larger --max-pages or fall back to popular search")


if __name__ == "__main__":
    main()
