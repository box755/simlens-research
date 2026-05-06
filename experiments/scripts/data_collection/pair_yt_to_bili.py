"""[DEPRECATED 2026-05-06] Auto-search Bilibili for likely-corresponding videos.

⚠️ v4.2.1 砍 Bilibili 採集（自動配對 22/49 全錯）。改用 YouTube timestamp
mentions 作 Anchor B（v4.2.1 §5.2.5 Group 1+ Anchor B）。本檔留作 future work
F-cross-lingual 的參考實作，**不再用於 SimLens 主 pipeline**。

Original docstring follows:

Strategy:
  Bilibili has many cross-platform mirror uploads of viral content (especially
  in the categories we picked: music covers, sports highlights, fail comp,
  cooking timelapse, animation). A title-keyword search on Bilibili will find
  candidates. We pull the top result and write a (yt_id, bvid) pair, then a
  human reviews the file before passing to collect_bilibili_danmaku.py.

  ⚠ Auto-pair quality is not 100%. The output file is meant to be a draft that
  a human eyeballs before download.

Bilibili search endpoint (public, no auth needed for first page):
  https://api.bilibili.com/x/web-interface/search/type
    ?search_type=video&keyword=<query>&order=totalrank&page=1

Returns items with `bvid`, `title` (HTML-stripped), `arcurl`, `duration`,
`pubdate`, `play` (view count), etc.

Usage:
  python scripts/data_collection/pair_yt_to_bili.py \
    --manifest data/raw_videos/manifest.anchor_b.jsonl \
    --out data/anchor_b/yt_to_bili_draft.jsonl

Then human reviews the draft, deletes wrong matches, renames to yt_to_bili.jsonl,
runs collect_bilibili_danmaku.py.
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

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)


def parse_bili_duration(s: str) -> int:
    """Bilibili duration format: 'M:SS' or 'H:MM:SS'. Returns seconds."""
    parts = [int(x) for x in s.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0


def search_bilibili(keyword: str) -> list[dict]:
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
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    if data.get("code") != 0:
        return []
    return data.get("data", {}).get("result", []) or []


def html_strip(text: str) -> str:
    return re.sub(r"</?em[^>]*>", "", text)


def best_candidate(yt_video: dict, candidates: list[dict],
                   tolerance_sec: int = 30) -> dict | None:
    yt_dur = yt_video["duration_sec"]
    scored = []
    for c in candidates:
        try:
            dur = parse_bili_duration(c.get("duration", "0:00"))
        except Exception:
            dur = 0
        if dur == 0:
            continue
        if abs(dur - yt_dur) > tolerance_sec:
            continue
        scored.append((c.get("play", 0), c, dur))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    play, cand, dur = scored[0]
    cand["_yt_dur"] = yt_dur
    cand["_bili_dur"] = dur
    return cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--duration-tolerance", type=int, default=30)
    args = ap.parse_args()

    with Path(args.manifest).open() as f:
        videos = [json.loads(l) for l in f]
    print(f"Pairing {len(videos)} YouTube videos to Bilibili candidates")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_paired = 0
    n_unpaired = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for v in videos:
            yt_id = v["id"]
            title = v["title"]
            try:
                results = search_bilibili(title)
            except Exception as e:
                print(f"  {yt_id}: search failed: {type(e).__name__}: {e}")
                n_unpaired += 1
                continue
            if not results:
                short_q = " ".join(title.split()[:3])
                if short_q != title:
                    try:
                        results = search_bilibili(short_q)
                    except Exception:
                        results = []
            cand = best_candidate(v, results, tolerance_sec=args.duration_tolerance)
            if cand is None:
                n_unpaired += 1
                rec = {
                    "yt_id": yt_id,
                    "yt_title": title,
                    "bvid": None,
                    "_note": "no Bilibili candidate within duration tolerance",
                }
            else:
                n_paired += 1
                rec = {
                    "yt_id": yt_id,
                    "yt_title": title,
                    "yt_duration_sec": v["duration_sec"],
                    "bvid": cand["bvid"],
                    "bili_title": html_strip(cand.get("title", "")),
                    "bili_duration_sec": cand["_bili_dur"],
                    "bili_play": cand.get("play"),
                    "bili_pubdate": cand.get("pubdate"),
                }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            print(f"  [{n_paired+n_unpaired}/{len(videos)}] {yt_id}: "
                  f"{'OK ' + cand['bvid'] if cand else 'NO MATCH'}")
            time.sleep(args.sleep)

    print(f"\nSummary: paired={n_paired} unpaired={n_unpaired}")
    print(f"\nNext step:")
    print(f"  1. Manually review {args.out} — delete or correct wrong matches")
    print(f"  2. Save as data/anchor_b/yt_to_bili.jsonl (drop entries with bvid=null)")
    print(f"  3. Run: python scripts/data_collection/collect_bilibili_danmaku.py "
          f"--pairs data/anchor_b/yt_to_bili.jsonl "
          f"--out-dir data/anchor_b/danmaku_timestamps")


if __name__ == "__main__":
    main()
