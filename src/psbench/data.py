"""Dataset loading and standardisation.

After `prepare`, every dataset has the same layout:
    obs["perturbation"]  canonical label, "control" or "A" or "A+B" (genes sorted)
    obs["n_perts"]       number of targeted genes (0 for control)
    obs["batch"]         batch / gem group
    obs["guide"]         guide label
    obs["size_factor"]   library size relative to the median
    layers["counts"]     raw counts (when the source provides them)
    X                    log1p(CP10k)
    uns["has_counts"]    whether layers["counts"] is real count data
"""
from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp

log = logging.getLogger(__name__)

COMBO_SEP = "+"
CONTROL = "control"
_CONTROL_TOKENS = {"ctrl", "control", "nt", "non-targeting", "non_targeting", "negctrl"}
_COUNT_LAYERS = ("counts", "raw_counts", "count")


def load_pertpy(loader: str) -> ad.AnnData:
    import pertpy as pt

    namespace = getattr(pt, "dt", None) or getattr(pt, "data")
    return getattr(namespace, loader)()


def parse_perturbation(name: str, sep: str | None = COMBO_SEP) -> tuple[str, ...]:
    """Split a raw label into its targeted genes, dropping control tokens."""
    tokens = str(name).split(sep) if sep else [str(name)]
    genes = {t.strip() for t in tokens if t.strip() and t.strip().lower() not in _CONTROL_TOKENS}
    return tuple(sorted(genes))


def canonical_name(genes: tuple[str, ...]) -> str:
    return COMBO_SEP.join(genes) if genes else CONTROL


def _looks_like_counts(M, n: int = 200_000) -> bool:
    vals = M.data if sp.issparse(M) else np.asarray(M).ravel()
    vals = vals[:n]
    return vals.size > 0 and vals.min() >= 0 and bool(np.allclose(vals, np.round(vals)))


def get_counts(adata: ad.AnnData):
    """Return (csr matrix, is_count_data). Searches X, count layers, then raw."""
    candidates = [adata.X] + [adata.layers[k] for k in _COUNT_LAYERS if k in adata.layers]
    if adata.raw is not None and adata.raw.X.shape == adata.shape:
        candidates.append(adata.raw.X)
    for M in candidates:
        if _looks_like_counts(M):
            return sp.csr_matrix(M, dtype=np.float32), True
    return sp.csr_matrix(adata.X, dtype=np.float32), False


def prepare(adata: ad.AnnData, ds: dict, pp: dict, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    sep = ds.get("combo_sep")

    sym = ds.get("var_symbol_col")
    if sym and sym in adata.var:
        adata.var_names = adata.var[sym].astype(str).values
    adata.var_names_make_unique()

    raw = adata.obs[ds["perturbation_col"]].astype(str)
    lut = {v: parse_perturbation(v, sep) for v in raw.unique()}
    genes = raw.map(lut)
    names = genes.map(canonical_name)
    n_perts = genes.map(len).to_numpy()

    var_set = set(adata.var_names)
    missing = {n for n, g in zip(names, genes) if any(x not in var_set for x in g)}
    if missing:
        log.warning("dropping %d perturbations whose target is not in var_names", len(missing))

    groups = names.groupby(names.to_numpy()).indices
    if CONTROL not in groups:
        raise ValueError(
            f"no control cells found in column {ds['perturbation_col']!r}; "
            "run `psbench inspect` and check the control label handling"
        )

    eligible = sorted(
        p for p, idx in groups.items()
        if p != CONTROL and p not in missing and len(idx) >= pp["min_cells_per_perturbation"]
    )
    max_p = ds.get("max_perturbations")
    if max_p and len(eligible) > max_p:
        eligible = sorted(rng.choice(eligible, max_p, replace=False).tolist())

    keep = []
    for p in eligible + [CONTROL]:
        idx = groups[p]
        limit = pp["max_control_cells"] if p == CONTROL else pp["max_cells_per_perturbation"]
        if len(idx) > limit:
            idx = rng.choice(idx, limit, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))

    sub = adata[keep].copy()
    sub.obs["perturbation"] = names.to_numpy()[keep]
    sub.obs["n_perts"] = n_perts[keep]

    M, has_counts = get_counts(sub)
    if has_counts:
        total = np.asarray(M.sum(1)).ravel()
        sf = np.clip(total / np.median(total[total > 0]), 1e-3, None)
        sub.layers["counts"] = M.copy()
        sub.X = M.copy()
        sc.pp.normalize_total(sub, target_sum=1e4)
        sc.pp.log1p(sub)
    else:
        log.warning("no raw counts found; count-based methods fall back to log-normalised data")
        sf = np.ones(sub.n_obs)
        sub.X = M

    targets = {g for p in eligible for g in p.split(COMBO_SEP)}
    n_top = pp.get("n_top_genes")
    if n_top and n_top < sub.n_vars:
        sc.pp.highly_variable_genes(sub, n_top_genes=n_top, flavor="seurat")
        sel = sub.var["highly_variable"].to_numpy() | sub.var_names.isin(targets)
        sub = sub[:, sel].copy()

    sub.obs["size_factor"] = sf if len(sf) == sub.n_obs else sf[: sub.n_obs]
    bcol, gcol = ds.get("batch_col"), ds.get("guide_col")
    sub.obs["batch"] = sub.obs[bcol].astype(str).values if bcol in sub.obs else "0"
    sub.obs["guide"] = sub.obs[gcol].astype(str).values if gcol in sub.obs else sub.obs["perturbation"].values
    sub.var["is_target"] = sub.var_names.isin(targets)
    sub.uns["has_counts"] = bool(has_counts)
    sub.uns["dataset"] = ds.get("name", "unknown")
    return sub


def describe(adata: ad.AnnData, max_cols: int = 40) -> str:
    lines = [
        f"shape: {adata.n_obs} cells x {adata.n_vars} genes",
        f"X: {type(adata.X).__name__}, counts-like: {_looks_like_counts(adata.X)}",
        f"layers: {list(adata.layers.keys())}",
        f"raw: {adata.raw is not None}",
        f"obs columns: {list(adata.obs.columns)[:max_cols]}",
        f"var columns: {list(adata.var.columns)[:max_cols]}",
        f"var_names head: {list(adata.var_names[:5])}",
    ]
    for col in adata.obs.columns:
        if adata.obs[col].dtype == object or str(adata.obs[col].dtype) == "category":
            n = adata.obs[col].nunique()
            if n < 15:
                lines.append(f"  {col}: {adata.obs[col].value_counts().head(8).to_dict()}")
    return "\n".join(lines)
