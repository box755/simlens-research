"""Sample 8 personas from PersonaChat 8K via cosine similarity + MMR.

Spec: SimLens_Research_Plan_v4.1.md §2.3 + Proposal 2026-05-04 Slide 13:
  - Source: bavard/personachat_truecased (HuggingFace)
  - Embedding: OpenAI text-embedding-3-small  (matches SimTube IUI 2025)
  - Query: aggregated keywords across ALL training videos (centroid query)
  - Diversity: MMR with λ=0.4 over top-80 candidates → top 8

Usage:
  python scripts/persona/sample_personachat.py \
    --keywords-file data/personas/aggregated_keywords.txt \
    --out data/personas/personas_sampled.yaml \
    --seed 42

Requires OPENAI_API_KEY for the embedding step. The PersonaChat dataset
(8K personas, ~50KB) is fetched once via `datasets`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.env import load_env

load_env()


def load_personachat() -> list[str]:
    """Return list of UNIQUE persona descriptions (each = newline-joined sentences).

    Source: AlekseyKorshuk/persona-chat (HuggingFace, parquet format, no script).
      - Original `bavard/personachat_truecased` is script-based and incompatible
        with datasets >= 4.0 (which dropped script support).
      - AlekseyKorshuk version is the same Persona-Chat (Zhang et al., ACL 2018)
        data, parquet-formatted, 17,878 train + 1,000 val dialogs.
      - Each dialog row repeats its personality across many turns; dedup yields
        ~8K unique personas (matching the original ACL 2018 paper count).
    """
    from datasets import load_dataset  # type: ignore

    seen: set[str] = set()
    out: list[str] = []
    for split in ("train", "validation"):
        try:
            ds = load_dataset("AlekseyKorshuk/persona-chat", split=split)
        except Exception as e:
            print(f"  WARN: failed to load split={split}: {e}")
            continue
        for row in ds:
            sentences = row.get("personality") or []
            if not sentences:
                continue
            text = "\n".join(s.strip() for s in sentences if s.strip())
            if text and text not in seen:
                seen.add(text)
                out.append(text)
    return out


def embed_texts(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """Batch-embed via OpenAI. Requires OPENAI_API_KEY."""
    from openai import OpenAI  # type: ignore

    client = OpenAI()
    out: list[list[float]] = []
    batch = 200
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        resp = client.embeddings.create(model=model, input=chunk)
        out.extend(d.embedding for d in resp.data)
    return out


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def mmr_select(
    candidate_idx: list[int],
    candidate_embs: list[list[float]],
    query_emb: list[float],
    k: int,
    lam: float = 0.4,
) -> list[int]:
    """MMR selection. Returns k indices from candidate_idx, max diversity tradeoff."""
    selected: list[int] = []
    remaining = list(candidate_idx)
    rel = {i: cosine(candidate_embs[i], query_emb) for i in remaining}

    while remaining and len(selected) < k:
        if not selected:
            best = max(remaining, key=lambda i: rel[i])
            selected.append(best)
            remaining.remove(best)
            continue
        best_i, best_score = None, -1e9
        for i in remaining:
            redundancy = max(cosine(candidate_embs[i], candidate_embs[j]) for j in selected)
            score = lam * rel[i] - (1 - lam) * redundancy
            if score > best_score:
                best_i, best_score = i, score
        selected.append(best_i)
        remaining.remove(best_i)
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keywords-file", required=True, help="text file: aggregated keywords (one per line or space-sep)")
    ap.add_argument("--out", required=True, help="output yaml path")
    ap.add_argument("--top-n", type=int, default=80, help="candidate pool size before MMR")
    ap.add_argument("--k", type=int, default=8, help="number of personas to select")
    ap.add_argument("--lam", type=float, default=0.4, help="MMR lambda (Proposal 2026-05-04: 0.4)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-embeddings", default="data/personas/personachat_embeddings.json",
                    help="cache file for persona embeddings (avoid re-embedding 8K texts)")
    args = ap.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: OPENAI_API_KEY not set. This script needs OpenAI for embeddings.", file=sys.stderr)
        print("Hint: copy experiments/.env.example to .env and fill in OPENAI_API_KEY,", file=sys.stderr)
        print("then `export $(grep -v '^#' .env | xargs)` before running.", file=sys.stderr)
        sys.exit(2)

    random.seed(args.seed)

    print(f"[1/4] Loading PersonaChat 8K from HuggingFace...")
    personas = load_personachat()
    print(f"  -> {len(personas)} unique persona descriptions")

    cache_path = Path(args.cache_embeddings)
    if cache_path.exists():
        print(f"[2/4] Loading cached embeddings from {cache_path}")
        with cache_path.open() as f:
            cached = json.load(f)
        if cached["count"] != len(personas):
            print("  WARN: cache size mismatch — re-embedding")
            cache = None
        else:
            cache = cached["embeddings"]
    else:
        cache = None

    if cache is None:
        print(f"[2/4] Embedding {len(personas)} personas (OpenAI text-embedding-3-small)...")
        cache = embed_texts(personas)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w") as f:
            json.dump({"count": len(personas), "embeddings": cache}, f)
        print(f"  -> cached to {cache_path}")

    print(f"[3/4] Embedding aggregated keyword query...")
    query_text = Path(args.keywords_file).read_text().strip()
    if not query_text:
        print("ERROR: empty keywords file", file=sys.stderr)
        sys.exit(2)
    query_emb = embed_texts([query_text])[0]

    print(f"[4/4] Top-{args.top_n} candidates → MMR top-{args.k} (lambda={args.lam})...")
    sims = [(i, cosine(cache[i], query_emb)) for i in range(len(personas))]
    sims.sort(key=lambda x: x[1], reverse=True)
    top_n_idx = [i for i, _ in sims[: args.top_n]]
    selected = mmr_select(top_n_idx, cache, query_emb, k=args.k, lam=args.lam)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Auto-generated by sample_personachat.py — do NOT hand-edit",
             f"# Source: AlekseyKorshuk/persona-chat (HF parquet, datasets-4 compatible)",
             f"#   Original: Persona-Chat (Zhang et al., ACL 2018), {len(personas)} unique personas",
             f"# Method: OpenAI text-embedding-3-small + cosine + MMR",
             f"# Seed: {args.seed}, top_n: {args.top_n}, lambda: {args.lam}",
             f"# Query: {Path(args.keywords_file).name}",
             ""]
    for rank, idx in enumerate(selected, 1):
        pid = f"P{rank}"
        sim_to_query = cosine(cache[idx], query_emb)
        lines.append(f"{pid}:")
        lines.append(f"  source_idx: {idx}")
        lines.append(f"  query_similarity: {sim_to_query:.4f}")
        lines.append(f"  description: |")
        for sent in personas[idx].split("\n"):
            lines.append(f"    {sent}")
        lines.append(f"  expected_comment_count_range: null  # filled by claude_estimate_activity.py")
        lines.append("")
    out_path.write_text("\n".join(lines))
    print(f"\nWrote {out_path}")
    print("\nNext: run scripts/persona/claude_estimate_activity.py to fill expected_comment_count_range")


if __name__ == "__main__":
    main()
