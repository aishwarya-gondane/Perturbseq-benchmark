"""Power analysis.

`simulate_power` draws negative-binomial counts with per-gene mean and
dispersion estimated from the control cells, plants effects on a fraction of
genes, and records how many are recovered at FDR 0.05 as a function of the
number of perturbed cells and the effect size.

`downsample_recovery` is the empirical counterpart: DEGs found with all cells
of a real perturbation are treated as the reference, and recovery is measured
after subsampling its cells.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .data import CONTROL
from .effects import welch


def estimate_params(adata, control: str = CONTROL, min_mean: float = 0.05):
    """Per-gene mean (size-factor normalised) and NB dispersion from control cells."""
    ctrl = adata[(adata.obs["perturbation"].astype(str) == control).to_numpy()]
    Y = ctrl.layers["counts"]
    Y = Y.toarray() if sp.issparse(Y) else np.asarray(Y)
    norm = Y / ctrl.obs["size_factor"].to_numpy()[:, None]
    mu, var = norm.mean(0), norm.var(0, ddof=1)
    alpha = np.clip((var - mu) / np.maximum(mu, 1e-8) ** 2, 1e-2, 10.0)
    keep = mu >= min_mean
    return mu[keep], alpha[keep]


def _draw(mu, alpha, n, log_fc, sf_sd, rng):
    sf = rng.lognormal(-sf_sd ** 2 / 2, sf_sd, n)
    mean = sf[:, None] * mu[None, :] * np.exp(log_fc)[None, :]
    counts = rng.poisson(rng.gamma(1 / alpha, alpha * mean))
    total = np.maximum(counts.sum(1, keepdims=True), 1)
    return np.log1p(counts / total * 1e4)


def simulate_power(mu, alpha, n_cells_grid, lfc_grid, n_control=1000, frac_de=0.1,
                   n_rep=5, fdr=0.05, sf_sd=0.3, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_genes = len(mu)
    rows = []
    for n, lfc in itertools.product(n_cells_grid, lfc_grid):
        for rep in range(n_rep):
            de = rng.random(n_genes) < frac_de
            planted = np.where(de, rng.choice([-1, 1], n_genes) * lfc * np.log(2), 0.0)
            ctrl = _draw(mu, alpha, n_control, np.zeros(n_genes), sf_sd, rng)
            trt = _draw(mu, alpha, n, planted, sf_sd, rng)
            p = stats.ttest_ind(trt, ctrl, equal_var=False, axis=0).pvalue
            p = np.nan_to_num(p, nan=1.0)
            rej = multipletests(p, method="fdr_bh", alpha=fdr)[0]
            rows.append({
                "n_cells": n,
                "log2fc": lfc,
                "rep": rep,
                "power": float((rej & de).sum() / max(de.sum(), 1)),
                "observed_fdr": float((rej & ~de).sum() / max(rej.sum(), 1)),
            })
    return pd.DataFrame(rows)


def downsample_recovery(adata, fractions, n_rep=3, fdr=0.05, min_perts=1,
                        control: str = CONTROL, seed=0, max_perts=10) -> pd.DataFrame:
    """Recovery of full-data DEGs when only a fraction of the perturbed cells is kept."""
    rng = np.random.default_rng(seed)
    obs = adata.obs
    counts = obs.loc[obs["n_perts"] == 1, "perturbation"].astype(str).value_counts()
    perts = counts.index[:max_perts]
    labels = obs["perturbation"].astype(str).to_numpy()

    rows = []
    for pert in perts:
        full = welch(adata, pert, control)
        ref = set(full.loc[full["fdr"] < fdr, "gene"])
        if len(ref) < min_perts:
            continue
        pert_idx = np.flatnonzero(labels == pert)
        other_idx = np.flatnonzero(labels == control)
        for frac in fractions:
            for rep in range(n_rep):
                k = max(int(len(pert_idx) * frac), 3)
                keep = np.concatenate([rng.choice(pert_idx, k, replace=False), other_idx])
                res = welch(adata[np.sort(keep)], pert, control)
                found = set(res.loc[res["fdr"] < fdr, "gene"])
                rows.append({"perturbation": pert, "fraction": frac, "n_cells": k, "rep": rep,
                             "n_reference_deg": len(ref),
                             "recovery": len(ref & found) / len(ref)})
    return pd.DataFrame(rows)
