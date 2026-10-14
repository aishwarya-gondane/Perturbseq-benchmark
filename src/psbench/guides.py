"""Guide assignment methods and a ground-truth benchmark on simulated guide counts.

Real data rarely comes with ground truth for guide identity, so the accuracy
comparison uses simulated counts with a known cell x guide truth. On real
guide-count matrices the same methods can be compared by call rate and
pairwise agreement (see `summarize_calls` and `pairwise_agreement`).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.mixture import GaussianMixture


def _dense(counts) -> np.ndarray:
    return np.asarray(counts.todense()) if sp.issparse(counts) else np.asarray(counts)


def simulate_guide_counts(
    n_cells: int = 5000,
    n_guides: int = 50,
    moi: float = 1.5,
    signal_mean: float = 100.0,
    ambient_mean: float = 3.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Cell x guide UMI counts and the boolean truth matrix."""
    rng = np.random.default_rng(seed)
    n_per_cell = np.minimum(rng.poisson(moi, n_cells), n_guides)
    truth = np.zeros((n_cells, n_guides), dtype=bool)
    for i, k in enumerate(n_per_cell):
        if k:
            truth[i, rng.choice(n_guides, k, replace=False)] = True

    depth = rng.lognormal(0, 0.5, n_cells)[:, None]
    guide_bias = rng.lognormal(0, 0.7, n_guides)[None, :]
    lam = (truth * signal_mean + ambient_mean * guide_bias) * depth
    lam = lam * rng.gamma(2.0, 0.5, size=lam.shape)
    return rng.poisson(lam), truth


def assign_threshold(counts, min_umi: int = 5) -> np.ndarray:
    """A guide is present when its UMI count reaches a fixed threshold."""
    return _dense(counts) >= min_umi


def assign_ratio(counts, min_umi: int = 3, min_frac: float = 0.2) -> np.ndarray:
    """Fixed threshold plus a minimum share of the cell's guide UMIs."""
    c = _dense(counts)
    frac = c / np.maximum(c.sum(1, keepdims=True), 1)
    return (c >= min_umi) & (frac >= min_frac)


def assign_top(counts, min_umi: int = 5) -> np.ndarray:
    """Each cell keeps only its highest-count guide, if it passes a threshold."""
    c = _dense(counts)
    out = np.zeros(c.shape, dtype=bool)
    top = c.argmax(1)
    rows = np.arange(c.shape[0])
    out[rows, top] = c[rows, top] >= min_umi
    return out


def assign_mixture(
    counts, min_umi: int = 2, posterior: float = 0.5, min_separation: float = 1.0, seed: int = 0
) -> np.ndarray:
    """Per-guide two-component Gaussian mixture on log1p counts.

    A cell is called positive when its posterior for the high component passes
    `posterior`. Guides whose components are not separated by `min_separation`
    log units (nothing to call) are left empty.
    """
    c = _dense(counts)
    out = np.zeros(c.shape, dtype=bool)
    for j in range(c.shape[1]):
        x = np.log1p(c[:, j])
        if (c[:, j] > 0).sum() < 10 or x.std() == 0:
            continue
        gm = GaussianMixture(2, random_state=seed).fit(x[:, None])
        hi, lo = int(np.argmax(gm.means_.ravel())), int(np.argmin(gm.means_.ravel()))
        if gm.means_[hi, 0] - gm.means_[lo, 0] < min_separation:
            continue
        out[:, j] = (gm.predict_proba(x[:, None])[:, hi] >= posterior) & (c[:, j] >= min_umi)
    return out


METHODS = {
    "threshold": assign_threshold,
    "ratio": assign_ratio,
    "top_guide": assign_top,
    "mixture": assign_mixture,
}


def evaluate_assignment(pred: np.ndarray, truth: np.ndarray) -> dict:
    tp = (pred & truth).sum()
    precision = tp / max(pred.sum(), 1)
    recall = tp / max(truth.sum(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "exact_cell_match": float((pred == truth).all(1).mean()),
    }


def summarize_calls(pred: np.ndarray) -> dict:
    n = pred.sum(1)
    return {
        "unassigned": float((n == 0).mean()),
        "singlet": float((n == 1).mean()),
        "multiplet": float((n >= 2).mean()),
    }


def pairwise_agreement(calls: dict[str, np.ndarray]) -> pd.DataFrame:
    """Jaccard index between the positive cell-guide pairs of each method pair."""
    rows = []
    for a, b in itertools.combinations(calls, 2):
        inter = (calls[a] & calls[b]).sum()
        union = (calls[a] | calls[b]).sum()
        rows.append({"method_a": a, "method_b": b, "jaccard": float(inter / max(union, 1))})
    return pd.DataFrame(rows)


def benchmark(cfg: dict, seed: int = 0) -> pd.DataFrame:
    rows = []
    for moi, signal in itertools.product(cfg["moi"], cfg["signal_mean"]):
        counts, truth = simulate_guide_counts(
            n_cells=cfg["n_cells"],
            n_guides=cfg["n_guides"],
            moi=moi,
            signal_mean=signal,
            ambient_mean=cfg["ambient_mean"],
            seed=seed,
        )
        for name, fn in METHODS.items():
            pred = fn(counts)
            rows.append(
                {"method": name, "moi": moi, "signal_mean": signal,
                 **evaluate_assignment(pred, truth), **summarize_calls(pred)}
            )
    return pd.DataFrame(rows)
