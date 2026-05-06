"""Real-World Anchor metrics for v4.2.3 §5.2.5 Group 1+ evaluation.

Anchor A (distributional match, 3 indicators):
  - Length KS-statistic ↓                              [Kolmogorov 1933, scipy]
  - Sentiment Wasserstein distance ↓                   [VADER ICWSM 2014]
  - Embedding Frechet Distance ↓                       [Ref 42 Heusel NeurIPS 2017]

Anchor B (per-video Hotspot Recall on high-mention test set, 2 indicators):
  - Hotspot Recall @ ±5s ↑                             [Ref 37 SoccerNet CVPR 2018]
  - Hotspot Precision @ ±5s ↑                          [Ref 37]

All deterministic — no LLM judge involved. Designed to break the Claude→Qwen→Qwen
closed-loop validation.

v4.2.3 changelog:
  - Removed v4.2.2 distribution-level Anchor B-1/B-2 metrics
    (mention_depth_ks, inter_gap_ks) — they remain available below as
    diagnostic / supplementary tools but are NO LONGER part of main result table
  - Restored per-video Hotspot Recall/Precision @ ±5s as primary Anchor B
    (was the v4.2 original design; v4.2.1/2 detoured because of test set
    mention sparsity, now solved by high-mention test set selection)

Usage (v4.2.3 main path):
  Anchor A:
    from anchor_metrics import length_ks, sentiment_wasserstein, embedding_frechet

  Anchor B (per-video):
    from anchor_metrics import (
        extract_timestamp_mentions,            # build ground truth
        kde_peaks,                              # ground-truth hotspots
        per_video_hotspot_recall_precision,    # per-video metric
        aggregate_hotspot_metrics,              # avg across test set
    )

  Distribution-level (legacy, available as supplementary):
    from anchor_metrics import mention_depth_ks, inter_gap_ks
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, Iterable

import numpy as np


# ============================================================================
# Anchor A: Distributional Match
# ============================================================================

def length_ks(simlens_corpus: list[str], real_corpus: list[str]) -> float:
    """Length distribution KS statistic. ↓ is better.

    Returns scipy.stats.ks_2samp statistic in [0, 1]; 0 = identical distributions.
    """
    from scipy.stats import ks_2samp  # type: ignore

    s_lens = [len(c) for c in simlens_corpus]
    r_lens = [len(c) for c in real_corpus]
    return float(ks_2samp(s_lens, r_lens).statistic)


def sentiment_wasserstein(
    simlens_corpus: list[str],
    real_corpus: list[str],
    classifier: Callable[[str], dict] | None = None,
) -> float:
    """Sentiment 1-Wasserstein distance over (neg, neu, pos) prob mass. ↓ better.

    classifier(text) -> {"neg": float, "neu": float, "pos": float}
    If classifier is None, uses VADER (light, deterministic).
    """
    if classifier is None:
        classifier = _vader_classifier()

    s_dist = _sentiment_dist(simlens_corpus, classifier)
    r_dist = _sentiment_dist(real_corpus, classifier)
    # 1-Wasserstein on a 3-point ordered support {neg=0, neu=1, pos=2}
    # (treating the labels as ordinal: more negative → 0, more positive → 2)
    s_cdf = np.cumsum([s_dist["neg"], s_dist["neu"], s_dist["pos"]])
    r_cdf = np.cumsum([r_dist["neg"], r_dist["neu"], r_dist["pos"]])
    return float(np.sum(np.abs(s_cdf - r_cdf)))  # discrete W1 over uniform-spaced support


def _sentiment_dist(corpus: list[str], classifier) -> dict[str, float]:
    counts = {"neg": 0, "neu": 0, "pos": 0}
    for text in corpus:
        scores = classifier(text)
        # majority class
        label = max(("neg", "neu", "pos"), key=lambda k: scores.get(k, 0.0))
        counts[label] += 1
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def _vader_classifier():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
    except ImportError:
        raise ImportError(
            "Install vaderSentiment for default sentiment classifier:\n"
            "  pip install vaderSentiment"
        )
    analyzer = SentimentIntensityAnalyzer()

    def classify(text: str) -> dict[str, float]:
        scores = analyzer.polarity_scores(text)
        return {"neg": scores["neg"], "neu": scores["neu"], "pos": scores["pos"]}

    return classify


# Style Marker χ² removed in v4.2.2 — handcrafted feature set (abbrev / emoji /
# caps / question) lacked direct literature support and reviewers would likely
# question its provenance. Length KS + Sentiment Wasserstein + Embedding
# Frechet provide sufficient text-distribution coverage with strong citations.


# ----- Embedding Frechet Distance -----

def embedding_frechet(
    simlens_corpus: list[str],
    real_corpus: list[str],
    embed_fn: Callable[[list[str]], list[list[float]]],
    sample_size: int = 5000,
    seed: int = 42,
) -> float:
    """Frechet (W2) distance between embedding distributions. ↓ better.

    Sample `sample_size` from each corpus, embed, treat each set as a
    multivariate Gaussian N(μ, Σ), compute Frechet distance:
      FD = ||μ_S − μ_R||² + Tr(Σ_S + Σ_R − 2 (Σ_S Σ_R)^½)

    Returns scaled distance (root of the squared formulation, more interpretable).
    """
    rng = np.random.default_rng(seed)
    s_sample = _sample(simlens_corpus, sample_size, rng)
    r_sample = _sample(real_corpus, sample_size, rng)

    s_emb = np.asarray(embed_fn(s_sample), dtype=np.float64)
    r_emb = np.asarray(embed_fn(r_sample), dtype=np.float64)

    mu_s, mu_r = s_emb.mean(axis=0), r_emb.mean(axis=0)
    sig_s = np.cov(s_emb, rowvar=False)
    sig_r = np.cov(r_emb, rowvar=False)

    diff = mu_s - mu_r
    mean_term = float(diff @ diff)

    # sqrt of product via SVD (avoid scipy.linalg.sqrtm complex artifacts)
    prod = sig_s @ sig_r
    eigvals = np.linalg.eigvals(prod)
    # take real parts; clip negatives (numerical noise) to 0
    sqrt_trace = float(np.sum(np.sqrt(np.clip(eigvals.real, 0, None))))

    fd_squared = mean_term + float(np.trace(sig_s) + np.trace(sig_r) - 2 * sqrt_trace)
    return math.sqrt(max(0.0, fd_squared))


def _sample(items: list[str], n: int, rng: np.random.Generator) -> list[str]:
    if len(items) <= n:
        return items
    idx = rng.choice(len(items), size=n, replace=False)
    return [items[i] for i in idx]


# ============================================================================
# Anchor B: Per-Video Hotspot Recall @ ±5s
# ============================================================================

def extract_timestamp_mentions(
    text: str,
    video_duration_sec: float,
) -> list[float]:
    """Extract timestamp mentions in seconds from a comment text.

    Matches "MM:SS" or "M:SS" (and "@MM:SS"). Filters out values exceeding
    video_duration_sec (likely clock-time false positives like "3:14 PM").
    """
    pattern = re.compile(r"@?(\d{1,2}):(\d{2})(?::(\d{2}))?")
    out: list[float] = []
    for m in pattern.finditer(text):
        try:
            mm = int(m.group(1))
            ss = int(m.group(2))
            hh = int(m.group(3)) if m.group(3) else 0
            if hh:
                seconds = hh * 3600 + mm * 60 + ss
            else:
                seconds = mm * 60 + ss
        except ValueError:
            continue
        if 0 <= seconds <= video_duration_sec + 5:  # tolerance
            out.append(float(seconds))
    return out


def kde_peaks(
    timestamps: list[float],
    video_duration_sec: float,
    bandwidth: float = 3.0,
    top_k: int | None = None,
    min_separation_sec: float = 5.0,
) -> list[float]:
    """KDE peak detection on a list of timestamps.

    Returns the top-k local maxima of the Gaussian KDE on a 1-second grid
    over [0, video_duration_sec]. If top_k is None, default to ⌈duration/30⌉.

    bandwidth=3.0 default chosen so closely-clustered timestamps merge into
    one peak rather than fragmenting (test_kde_peaks empirical tuning).
    min_separation_sec enforces minimum spacing between returned peaks.
    """
    if not timestamps:
        return []
    if top_k is None:
        top_k = max(1, math.ceil(video_duration_sec / 30.0))

    grid = np.arange(0, video_duration_sec + 0.5, 1.0)
    ts_arr = np.asarray(timestamps, dtype=np.float64)
    diff = grid[:, None] - ts_arr[None, :]
    density = np.exp(-0.5 * (diff / bandwidth) ** 2).sum(axis=1)
    density /= bandwidth * math.sqrt(2 * math.pi)

    # Local maxima: density >= both neighbors (allow plateaus by using >=
    # forward + > backward heuristic, then dedup by min_separation).
    peaks_idx: list[int] = []
    n = len(density)
    for i in range(n):
        left_ok = (i == 0) or density[i] >= density[i - 1]
        right_ok = (i == n - 1) or density[i] >= density[i + 1]
        # Require at least one strict inequality to avoid flat regions
        strict = ((i > 0 and density[i] > density[i - 1])
                  or (i < n - 1 and density[i] > density[i + 1]))
        if left_ok and right_ok and strict:
            peaks_idx.append(i)
    if not peaks_idx:
        peaks_idx = [int(np.argmax(density))]

    # Sort by density desc, then enforce min_separation
    peaks_idx.sort(key=lambda i: -density[i])
    chosen: list[int] = []
    for idx in peaks_idx:
        if all(abs(grid[idx] - grid[c]) >= min_separation_sec for c in chosen):
            chosen.append(idx)
            if len(chosen) >= top_k:
                break

    return sorted(float(grid[i]) for i in chosen)


def per_video_hotspot_recall_precision(
    real_hotspots: list[float],
    predicted_timestamps: list[float],
    tolerance_sec: float = 5.0,
) -> tuple[float, float]:
    """For one video, compute (recall, precision) of predicted vs real hotspots.

    Recall = (# real hotspots within tolerance of any prediction) / # real hotspots
    Precision = (# predictions within tolerance of any real hotspot) / # predictions

    Both use greedy matching: each real hotspot can be claimed by at most one
    prediction (within tolerance), and vice versa.
    """
    if not real_hotspots and not predicted_timestamps:
        return (1.0, 1.0)
    if not real_hotspots:
        return (0.0, 0.0)
    if not predicted_timestamps:
        return (0.0, 0.0)

    # Greedy matching
    matched_pred: set[int] = set()
    matched_real: set[int] = set()
    for ri, r in enumerate(real_hotspots):
        best: int | None = None
        best_diff = tolerance_sec + 1
        for pi, p in enumerate(predicted_timestamps):
            if pi in matched_pred:
                continue
            d = abs(p - r)
            if d <= tolerance_sec and d < best_diff:
                best, best_diff = pi, d
        if best is not None:
            matched_pred.add(best)
            matched_real.add(ri)

    recall = len(matched_real) / len(real_hotspots)
    precision = len(matched_pred) / len(predicted_timestamps)
    return (recall, precision)


def aggregate_hotspot_metrics(
    per_video_results: list[tuple[float, float]],
) -> dict[str, float]:
    """Average per-video (recall, precision) across the test set."""
    if not per_video_results:
        return {"recall": 0.0, "precision": 0.0, "n": 0}
    recalls = [r for r, _ in per_video_results]
    precisions = [p for _, p in per_video_results]
    return {
        "recall": float(np.mean(recalls)),
        "precision": float(np.mean(precisions)),
        "n": len(per_video_results),
    }


# ============================================================================
# Anchor B (supplementary, distribution-level): when per-video mentions are
# too sparse for reliable per-video hotspot recall, fall back to comparing
# the *aggregate* distribution of timestamp mentions across the test set
# vs SimLens's aggregate predicted timestamps.
# ============================================================================

def mention_depth_distribution(
    mentions_per_video: dict[str, list[float]],
    durations: dict[str, float],
) -> list[float]:
    """For each (video, mention) pair, return mention_sec / video_duration ∈ [0, 1].

    Returns flat list of normalized depths across all videos.
    """
    out: list[float] = []
    for vid, mentions in mentions_per_video.items():
        dur = durations.get(vid)
        if not dur or dur <= 0:
            continue
        for m in mentions:
            depth = m / dur
            if 0 <= depth <= 1:
                out.append(depth)
    return out


def mention_depth_ks(
    real_mentions: dict[str, list[float]],
    predicted_timestamps: dict[str, list[float]],
    durations: dict[str, float],
) -> float:
    """KS statistic between real-mention depth distribution and SimLens-predicted
    depth distribution. ↓ better.

    Both inputs are dict[video_id -> list of timestamp seconds]; durations is
    dict[video_id -> duration_sec]. Aggregates across all videos in the test set.
    """
    from scipy.stats import ks_2samp  # type: ignore

    real_depths = mention_depth_distribution(real_mentions, durations)
    pred_depths = mention_depth_distribution(predicted_timestamps, durations)
    if not real_depths or not pred_depths:
        return 1.0
    return float(ks_2samp(real_depths, pred_depths).statistic)


def inter_gap_distribution(
    timestamps_per_video: dict[str, list[float]],
) -> list[float]:
    """Within each video, sort timestamps and compute consecutive gaps.

    Returns flat list of gaps (in seconds) across all videos.
    """
    out: list[float] = []
    for vid, ts in timestamps_per_video.items():
        ts_sorted = sorted(ts)
        for i in range(len(ts_sorted) - 1):
            gap = ts_sorted[i + 1] - ts_sorted[i]
            if gap > 0:
                out.append(gap)
    return out


def inter_gap_ks(
    real_mentions: dict[str, list[float]],
    predicted_timestamps: dict[str, list[float]],
) -> float:
    """KS statistic between real-mention inter-gap distribution and
    SimLens-predicted inter-gap distribution. ↓ better.

    Captures whether SimLens places timestamps at human-like intervals.
    """
    from scipy.stats import ks_2samp  # type: ignore

    real_gaps = inter_gap_distribution(real_mentions)
    pred_gaps = inter_gap_distribution(predicted_timestamps)
    if not real_gaps or not pred_gaps:
        return 1.0
    return float(ks_2samp(real_gaps, pred_gaps).statistic)


def mention_density_ratio(
    real_mentions: dict[str, list[float]],
    predicted_timestamps: dict[str, list[float]],
) -> dict[str, float]:
    """Compare mentions per video (real) vs predictions per video (SimLens).

    Returns {real_avg, pred_avg, ratio = pred / real}.
    """
    n_real = sum(len(v) for v in real_mentions.values())
    n_pred = sum(len(v) for v in predicted_timestamps.values())
    n_videos_real = max(1, sum(1 for v in real_mentions.values() if v))
    n_videos_pred = max(1, sum(1 for v in predicted_timestamps.values() if v))
    real_avg = n_real / n_videos_real
    pred_avg = n_pred / n_videos_pred
    return {
        "real_avg_per_video": real_avg,
        "pred_avg_per_video": pred_avg,
        "density_ratio": pred_avg / real_avg if real_avg > 0 else 0.0,
    }
