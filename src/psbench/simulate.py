"""Small synthetic Perturb-seq dataset with planted effects.

Used by the test suite and for smoke-testing the workflow without a download.
Each target gene is knocked down and shifts a set of other genes; each pair of
targets is additive plus a planted interaction on a separate set of genes.
"""
from __future__ import annotations

from itertools import combinations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .data import CONTROL, COMBO_SEP, canonical_name


def toy_dataset(
    n_genes: int = 200,
    n_targets: int = 6,
    n_control: int = 400,
    cells_per_pert: int = 120,
    n_de: int = 20,
    n_interaction: int = 10,
    combos: bool = True,
    alpha: float = 0.1,
    seed: int = 0,
) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    var_names = [f"gene{i}" for i in range(n_genes)]
    base = rng.lognormal(1.5, 0.8, n_genes)
    non_target = np.arange(n_targets, n_genes)

    single_eff, planted_de = {}, {}
    for t in range(n_targets):
        eff = np.zeros(n_genes)
        eff[t] = np.log(0.2)
        de = rng.choice(non_target, n_de, replace=False)
        eff[de] = rng.choice([-1, 1], n_de) * np.log(2) * rng.uniform(0.7, 1.5, n_de)
        single_eff[f"gene{t}"] = eff
        planted_de[f"gene{t}"] = [var_names[i] for i in de]

    conds = {CONTROL: np.zeros(n_genes)}
    conds.update(single_eff)
    planted_int = {}
    if combos:
        for a, b in combinations(sorted(single_eff), 2):
            name = canonical_name(tuple(sorted((a, b))))
            eff = single_eff[a] + single_eff[b]
            idx = rng.choice(non_target, n_interaction, replace=False)
            eff[idx] += rng.choice([-1, 1], n_interaction) * np.log(2) * 1.5
            conds[name] = eff
            planted_int[name] = [var_names[i] for i in idx]

    labels, effs = [], []
    for name, eff in conds.items():
        n = n_control if name == CONTROL else cells_per_pert
        labels += [name] * n
        effs.append(np.tile(eff, (n, 1)))
    effs = np.vstack(effs)
    n_cells = len(labels)

    sf = rng.lognormal(0, 0.3, n_cells)
    mu = sf[:, None] * base[None, :] * np.exp(effs)
    lam = rng.gamma(1 / alpha, alpha * mu)
    counts = rng.poisson(lam).astype(np.float32)

    total = counts.sum(1)
    lognorm = np.log1p(counts / np.maximum(total, 1)[:, None] * 1e4).astype(np.float32)

    labels = np.array(labels)
    obs = pd.DataFrame(
        {
            "perturbation": labels,
            "n_perts": [0 if p == CONTROL else p.count(COMBO_SEP) + 1 for p in labels],
            "batch": rng.choice(["0", "1"], n_cells),
            "size_factor": total / np.median(total),
        },
        index=[f"cell{i}" for i in range(n_cells)],
    )
    obs["guide"] = [f"{p}_g{rng.integers(2)}" for p in labels]

    adata = ad.AnnData(X=lognorm, obs=obs, var=pd.DataFrame(index=var_names))
    adata.layers["counts"] = sp.csr_matrix(counts)
    adata.uns["has_counts"] = True
    adata.uns["dataset"] = "toy"
    adata.uns["planted_de"] = planted_de
    adata.uns["planted_interaction"] = planted_int
    return adata
