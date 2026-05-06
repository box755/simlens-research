"""Fetch danmaku TIMESTAMPS for v4.2.2 Anchor B-2 (Bilibili popular short videos).

Reads bili_popular_manifest.jsonl (output of collect_bilibili_popular.py),
fetches the danmaku XML for each cid, parses ONLY the time field of each
<d> element, writes per-video timestamp list.

Privacy: danmaku text content is deliberately discarded; only second-level
timestamps are stored.

Usage:
  python scripts/data_collection/collect_bilibili_popular_danmaku.py \
    --manifest data/anchor_b2/bili_popular_manifest.jsonl \
    --out-dir data/anchor_b2/danmaku_timestamps
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)


def fetch_danmaku_xml(cid: int) -> bytes:
    url = f"https://comment.bilibili.com/{cid}.xml"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def parse_timestamps_only(xml_bytes: bytes) -> list[float]:
    """Parse <d p="time,..."> entries; return only time field as float seconds.

    Format of `p` attribute: time,mode,size,color,timestamp,pool,user,id

    Bilibili's comment.bilibili.com/{cid}.xml endpoint serves the XML payload
    deflate-compressed (raw zlib, no gzip header). We try multiple decompress
    strategies before giving up.
    """
    import zlib

    text: str | None = None
    # Try as plain UTF-8 first (some endpoints serve plaintext)
    try:
        candidate = xml_bytes.decode("utf-8")
        if "<i>" in candidate or "<d " in candidate:
            text = candidate
    except UnicodeDecodeError:
        pass

    # Try raw deflate (no zlib header)
    if text is None:
        try:
            decompressed = zlib.decompress(xml_bytes, -zlib.MAX_WBITS)
            text = decompressed.decode("utf-8", errors="ignore")
        except zlib.error:
            pass

    # Try standard zlib (with header)
    if text is None:
        try:
            decompressed = zlib.decompress(xml_bytes)
            text = decompressed.decode("utf-8", errors="ignore")
        except zlib.error:
            pass

    if text is None or "<d " not in text:
        return []

    timestamps: list[float] = []
    try:
        root = ET.fromstring(text)
    except ET.ParseError as e:
        print(f"  XML parse error: {e}", file=sys.stderr)
        return []
    for d in root.findall("d"):
        p = d.get("p", "")
        if not p:
            continue
        try:
            t = float(p.split(",", 1)[0])
            if t >= 0:
                timestamps.append(t)
        except ValueError:
            continue
    return timestamps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="bili_popular_manifest.jsonl from collect_bilibili_popular.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()

    with Path(args.manifest).open(encoding="utf-8") as f:
        videos = [json.loads(l) for l in f]
    print(f"Fetching danmaku for {len(videos)} Bilibili videos")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_done = n_skip = n_fail = 0
    total_ts = 0
    for i, v in enumerate(videos, 1):
        bvid = v["bvid"]
        cid = v.get("cid")
        if not cid:
            print(f"  [{i}/{len(videos)}] {bvid}: no cid in manifest, skip")
            n_fail += 1
            continue

        out_path = out_dir / f"{bvid}.json"
        if out_path.exists():
            n_skip += 1
            continue

        try:
            xml_bytes = fetch_danmaku_xml(cid)
        except Exception as e:
            print(f"  [{i}/{len(videos)}] {bvid}: fetch failed: {type(e).__name__}: {e}")
            n_fail += 1
            continue

        timestamps = parse_timestamps_only(xml_bytes)
        if not timestamps:
            print(f"  [{i}/{len(videos)}] {bvid}: 0 timestamps parsed")
            n_fail += 1
            continue

        rec = {
            "bvid": bvid,
            "cid": cid,
            "duration_sec": v.get("duration_sec"),
            "title_truncated": (v.get("title") or "")[:50],
            "n_danmaku": len(timestamps),
            "timestamps": timestamps,
            "_disclaimer": (
                "Only second-level timestamps stored. Danmaku text content is "
                "deliberately NOT recorded for privacy/legal compliance "
                "(SimLens v4.2.2 §3.2 Step 1.4 Anchor B-2)."
            ),
        }
        out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        n_done += 1
        total_ts += len(timestamps)
        print(f"  [{i}/{len(videos)}] {bvid}: {len(timestamps)} timestamps")
        time.sleep(args.sleep)

    print(f"\nSummary: done={n_done} skip={n_skip} fail={n_fail}")
    print(f"  total timestamps collected: {total_ts}")
    if n_done > 0:
        print(f"  avg per video: {total_ts / n_done:.1f}")


if __name__ == "__main__":
    main()
