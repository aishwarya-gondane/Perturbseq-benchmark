"""Perturbation-effect estimators, each compared against non-targeting controls.

All per-perturbation methods return a long table with the columns in COLUMNS.

    nb_glm   negative-binomial GLM on counts with a size-factor offset and a
             batch covariate (OLS on log-normalised data if counts are absent)
    mixed    linear mixed model on log-normalised data, random intercept per
             group (default: batch), fitted on a gene subset for speed
    welch    Welch t-test on log-normalised data, the simple baseline
    mixscape pertpy Mixscape; cells are first classified as KO / non-perturbed
             and the KO cells are then compared with control
"""
from __future__ import annotations

import inspect
import logging
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .data import COMBO_SEP, CONTROL

log = logging.getLogger(__name__)

COLUMNS = ["perturbation", "gene", "log2fc", "stat", "pvalue", "fdr", "method"]
LN2 = np.log(2)


def single_perturbations(adata) -> list[str]:
    obs = adata.obs
    return sorted(obs.loc[obs["n_perts"] == 1, "perturbation"].astype(str).unique())


def _dense(M) -> np.ndarray:
    return M.toarray() if sp.issparse(M) else np.asarray(M)


def _subset(adata, pert: str, control: str):
    labels = adata.obs["perturbation"].astype(str)
    sub = adata[labels.isin([pert, control]).to_numpy()]
    treated = (sub.obs["perturbation"].astype(str) == pert).to_numpy()
    return sub, treated


def _bh(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return out


def _table(pert, method, genes, lfc, stat, p) -> pd.DataFrame:
    return pd.DataFrame(
        {"perturbation": pert, "gene": list(genes), "log2fc": lfc, "stat": stat,
         "pvalue": p, "fdr": _bh(p), "method": method}
    )[COLUMNS]


def _design(sub, treated, covariate):
    cols = [np.ones(len(treated)), treated.astype(float)]
    if covariate and covariate in sub.obs and sub.obs[covariate].astype(str).nunique() > 1:
        dummies = pd.get_dummies(sub.obs[covariate].astype(str), drop_first=True)
        cols.extend(dummies.to_numpy(float).T)
    return np.column_stack(cols)


def nb_glm(adata, pert, control=CONTROL, covariate="batch", min_frac=0.05, **_) -> pd.DataFrame:
    sub, treated = _subset(adata, pert, control)
    has_counts = bool(adata.uns.get("has_counts", True))
    Y = _dense(sub.layers["counts"] if has_counts else sub.X)
    design = _design(sub, treated, covariate)
    offset = np.log(sub.obs["size_factor"].to_numpy(float))
    ctrl = ~treated

    detected = np.maximum((Y[treated] > 0).mean(0), (Y[ctrl] > 0).mean(0))
    lfc = np.full(Y.shape[1], np.nan)
    stat = np.full(Y.shape[1], np.nan)
    pval = np.full(Y.shape[1], np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j in np.flatnonzero(detected >= min_frac):
            y = Y[:, j]
            try:
                if has_counts:
                    m, v = y[ctrl].mean(), y[ctrl].var(ddof=1)
                    alpha = max((v - m) / max(m, 1e-8) ** 2, 1e-2)
                    fit = sm.GLM(
                        y, design, offset=offset,
                        family=sm.families.NegativeBinomial(alpha=alpha),
                    ).fit()
                else:
                    fit = sm.OLS(y, design).fit()
                lfc[j] = fit.params[1] / LN2
                stat[j] = fit.tvalues[1]
                pval[j] = fit.pvalues[1]
            except Exception as exc:  # singular fits on near-empty genes
                log.debug("nb_glm failed for %s / %s: %s", pert, sub.var_names[j], exc)
    return _table(pert, "nb_glm", sub.var_names, lfc, stat, pval)


def welch(adata, pert, control=CONTROL, **_) -> pd.DataFrame:
    sub, treated = _subset(adata, pert, control)
    X = _dense(sub.X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = stats.ttest_ind(X[treated], X[~treated], equal_var=False, axis=0)
    lfc = (X[treated].mean(0) - X[~treated].mean(0)) / LN2
    return _table(pert, "welch", sub.var_names, lfc, res.statistic, res.pvalue)


def mixed(adata, pert, control=CONTROL, group_col="batch", n_genes=300, **_) -> pd.DataFrame:
    """y ~ treated + (1 | group). Tested genes: the target(s) plus the most variable."""
    sub, treated = _subset(adata, pert, control)
    X = _dense(sub.X)
    groups = sub.obs[group_col].astype(str).to_numpy()
    if len(np.unique(groups)) < 2:
        raise ValueError(f"mixed model needs at least two levels in {group_col!r}")

    names = np.asarray(sub.var_names)
    targets = [np.flatnonzero(names == g) for g in pert.split(COMBO_SEP)]
    chosen = set(np.argsort(X.var(0))[::-1][:n_genes].tolist())
    chosen.update(int(t[0]) for t in targets if len(t))

    exog = np.column_stack([np.ones(len(treated)), treated.astype(float)])
    lfc = np.full(X.shape[1], np.nan)
    stat = np.full(X.shape[1], np.nan)
    pval = np.full(X.shape[1], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j in sorted(chosen):
            try:
                fit = sm.MixedLM(X[:, j], exog, groups=groups).fit(reml=True, method=["lbfgs"])
                lfc[j] = fit.params[1] / LN2
                stat[j] = fit.tvalues[1]
                pval[j] = fit.pvalues[1]
            except Exception as exc:
                log.debug("mixed failed for %s / %s: %s", pert, names[j], exc)
    return _table(pert, "mixed", names, lfc, stat, pval)


PER_PERT_METHODS = {"nb_glm": nb_glm, "mixed": mixed, "welch": welch}


def method_kwargs(method: str, ec: dict) -> dict:
    if method == "nb_glm":
        return {"covariate": "batch", "min_frac": ec["min_frac_expressed"]}
    if method == "mixed":
        return {"group_col": ec.get("mixed_group_col", "batch"), "n_genes": ec["mixed_genes"]}
    return {}


def run_per_pert(adata, perts, method, control=CONTROL, **kw) -> pd.DataFrame:
    fn = PER_PERT_METHODS[method]
    frames = []
    for i, p in enumerate(perts, 1):
        try:
            frames.append(fn(adata, p, control, **kw))
        except Exception as exc:
            log.warning("%s failed for %s: %s", method, p, exc)
        if i % 10 == 0:
            log.info("%s: %d/%d perturbations", method, i, len(perts))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)


def _pert_key_arg(fn) -> str:
    """pertpy renamed `labels` to `pert_key` in some releases."""
    return "pert_key" if "pert_key" in inspect.signature(fn).parameters else "labels"


def mixscape(adata, perts=None, control=CONTROL, batch_col="batch", n_neighbors=20,
             min_de_genes=5, logfc_threshold=0.25, seed=0):
    """Run pertpy Mixscape on the single perturbations.

    Returns (effects, summary). `effects` compares the cells classified as KO
    with all control cells; `summary` holds the KO / non-perturbed split.
    """
    import pertpy as pt
    import scanpy as sc

    perts = list(perts) if perts is not None else single_perturbations(adata)
    labels = adata.obs["perturbation"].astype(str)
    a = adata[labels.isin(perts + [control]).to_numpy()].copy()
    a.obs["perturbation"] = a.obs["perturbation"].astype(str)

    split = batch_col if batch_col in a.obs and a.obs[batch_col].astype(str).nunique() > 1 else None
    ms = pt.tl.Mixscape()
    ms.perturbation_signature(
        a, pert_key="perturbation", control=control, split_by=split, n_neighbors=n_neighbors
    )
    kwargs = {
        _pert_key_arg(ms.mixscape): "perturbation",
        "control": control,
        "min_de_genes": min_de_genes,
        "logfc_threshold": logfc_threshold,
    }
    if "random_state" in inspect.signature(ms.mixscape).parameters:
        kwargs["random_state"] = seed
    ms.mixscape(adata=a, **kwargs)

    cls = a.obs["mixscape_class_global"].astype(str).to_numpy()
    lab = a.obs["perturbation"].to_numpy()
    X = _dense(a.X)
    is_ctrl = lab == control

    frames, summary = [], []
    for p in perts:
        in_p = lab == p
        ko = in_p & (cls == "KO")
        summary.append({"perturbation": p, "n_cells": int(in_p.sum()), "n_ko": int(ko.sum()),
                        "ko_fraction": float(ko.sum() / max(in_p.sum(), 1))})
        if ko.sum() < 3:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = stats.ttest_ind(X[ko], X[is_ctrl], equal_var=False, axis=0)
        lfc = (X[ko].mean(0) - X[is_ctrl].mean(0)) / LN2
        frames.append(_table(p, "mixscape", a.var_names, lfc, res.statistic, res.pvalue))
    effects = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    return effects, pd.DataFrame(summary)
