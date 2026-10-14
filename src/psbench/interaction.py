"""Interaction model for double perturbations (Norman 2019).

For every combination A+B, cells from {control, A, B, A+B} are fit per gene with

    expression ~ 1 + has_A + has_B + has_A:has_B

where has_A is true for cells carrying A (including A+B cells). The
interaction coefficient is the deviation of the double from the sum of the
single effects; it is tested with a t-test on the OLS estimate.

Per combination the pseudo-bulk profile deltas (mean minus control mean) are
also summarised, following the regression used in Norman et al.:

    delta_AB ~ c1 * delta_A + c2 * delta_B
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .data import COMBO_SEP, CONTROL


def _dense(M) -> np.ndarray:
    return M.toarray() if sp.issparse(M) else np.asarray(M)


def combos(adata) -> list[tuple[str, str, str]]:
    labels = set(adata.obs["perturbation"].astype(str).unique())
    out = []
    for lab in sorted(labels):
        parts = lab.split(COMBO_SEP)
        if len(parts) == 2 and all(p in labels for p in parts):
            out.append((parts[0], parts[1], lab))
    return out


def interaction_for_combo(adata, a: str, b: str, ab: str, control: str = CONTROL,
                          min_cells: int = 20):
    """Return (per-gene table, summary dict) or None if too few cells."""
    labels = adata.obs["perturbation"].astype(str).to_numpy()
    mask = np.isin(labels, [control, a, b, ab])
    lab = labels[mask]
    sizes = {c: int((lab == c).sum()) for c in (control, a, b, ab)}
    if min(sizes.values()) < min_cells:
        return None

    Y = _dense(adata.X[mask])
    has_a = np.isin(lab, [a, ab]).astype(float)
    has_b = np.isin(lab, [b, ab]).astype(float)
    D = np.column_stack([np.ones(len(lab)), has_a, has_b, has_a * has_b])
    inv = np.linalg.pinv(D.T @ D)
    beta = inv @ D.T @ Y
    resid = Y - D @ beta
    dof = len(lab) - 4
    s2 = (resid ** 2).sum(0) / dof
    se = np.sqrt(s2 * inv[3, 3])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta[3] / se, np.nan)
    p = 2 * stats.t.sf(np.abs(t), dof)
    ok = np.isfinite(p)
    fdr = np.full_like(p, np.nan)
    if ok.any():
        fdr[ok] = multipletests(p[ok], method="fdr_bh")[1]

    genes = pd.DataFrame({
        "combination": ab, "gene": adata.var_names, "interaction": beta[3],
        "single_a": beta[1], "single_b": beta[2], "pvalue": p, "fdr": fdr,
    })

    mean = {c: Y[lab == c].mean(0) for c in (control, a, b, ab)}
    d_a, d_b, d_ab = (mean[c] - mean[control] for c in (a, b, ab))
    coef, *_ = np.linalg.lstsq(np.column_stack([d_a, d_b]), d_ab, rcond=None)
    fitted = np.column_stack([d_a, d_b]) @ coef
    norm_ab = np.linalg.norm(d_ab)
    additive = d_a + d_b
    summary = {
        "combination": ab, "gene_a": a, "gene_b": b,
        "n_cells_a": sizes[a], "n_cells_b": sizes[b], "n_cells_ab": sizes[ab],
        "c1": float(coef[0]), "c2": float(coef[1]),
        "r2_dosage": float(1 - ((d_ab - fitted) ** 2).sum() / max((d_ab ** 2).sum(), 1e-12)),
        "norm_ab": float(norm_ab),
        "additive_error": float(np.linalg.norm(d_ab - additive) / max(norm_ab, 1e-12)),
        "magnitude_ratio": float(norm_ab / max(np.linalg.norm(additive), 1e-12)),
    }
    return genes, summary


def run_interactions(adata, control: str = CONTROL, min_cells: int = 20, fdr: float = 0.05,
                     synergy_ratio: float = 1.2, buffer_ratio: float = 0.8):
    gene_tables, summaries = [], []
    for a, b, ab in combos(adata):
        out = interaction_for_combo(adata, a, b, ab, control, min_cells)
        if out is None:
            continue
        genes, summary = out
        summary["n_interaction_genes"] = int((genes["fdr"] < fdr).sum())
        r = summary["magnitude_ratio"]
        summary["class"] = (
            "synergistic" if r > synergy_ratio else "buffering" if r < buffer_ratio else "additive"
        )
        summaries.append(summary)
        gene_tables.append(genes[genes["fdr"] < fdr])
    genes_all = pd.concat(gene_tables, ignore_index=True) if gene_tables else pd.DataFrame()
    return pd.DataFrame(summaries), genes_all
