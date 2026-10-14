"""Evaluation of effect estimators: on-target knockdown, DEG counts, null calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .data import CONTROL
from .effects import PER_PERT_METHODS, single_perturbations


def on_target_summary(res: pd.DataFrame, fdr: float = 0.05) -> pd.DataFrame:
    """Does each method see the knockdown of the targeted gene itself?"""
    t = res[res["gene"] == res["perturbation"]].copy()
    t["knockdown"] = (t["log2fc"] < 0) & (t["fdr"] < fdr)
    return (
        t.groupby("method")
        .agg(n_perts=("perturbation", "nunique"),
             knockdown_rate=("knockdown", "mean"),
             median_log2fc=("log2fc", "median"))
        .reset_index()
    )


def deg_summary(res: pd.DataFrame, fdr: float = 0.05) -> pd.DataFrame:
    n = (
        res.assign(sig=res["fdr"] < fdr)
        .groupby(["method", "perturbation"])["sig"].sum()
        .rename("n_deg").reset_index()
    )
    return n.groupby("method")["n_deg"].describe()[["count", "50%", "mean", "max"]].reset_index()


def null_calibration(adata, method: str, n_groups: int = 10, group_size: int = 100,
                     seed: int = 0, control: str = CONTROL, **kw) -> pd.DataFrame:
    """Split non-targeting cells into pseudo-perturbations and test them.

    Any signal here is a false positive. Reports the share of genes with
    p < 0.05, the number of genes called at FDR 0.05, and a KS statistic of
    the p-value distribution against uniform.
    """
    rng = np.random.default_rng(seed)
    labels = adata.obs["perturbation"].astype(str).to_numpy()
    ctrl_idx = np.flatnonzero(labels == control)
    group_size = min(group_size, max((len(ctrl_idx) - 50) // n_groups, 2))
    chosen = rng.permutation(ctrl_idx)[: n_groups * group_size]

    a = adata[ctrl_idx].copy()
    new = np.full(a.n_obs, control, dtype=object)
    pos = {c: i for i, c in enumerate(ctrl_idx)}
    for g in range(n_groups):
        for c in chosen[g * group_size:(g + 1) * group_size]:
            new[pos[c]] = f"null_{g}"
    a.obs["perturbation"] = new
    a.obs["n_perts"] = np.where(new == control, 0, 1)

    fn = PER_PERT_METHODS[method]
    rows = []
    for g in range(n_groups):
        res = fn(a, f"null_{g}", control, **kw)
        p = res["pvalue"].dropna().to_numpy()
        if p.size == 0:
            continue
        rows.append({
            "method": method,
            "group": g,
            "n_tested": int(p.size),
            "frac_p_lt_0.05": float((p < 0.05).mean()),
            "n_deg_fdr_0.05": int((res["fdr"] < 0.05).sum()),
            "ks_vs_uniform": float(stats.kstest(p, "uniform").statistic),
        })
    return pd.DataFrame(rows)
