import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

from psbench import data


def test_parse_perturbation_single_and_combo():
    assert data.parse_perturbation("KLF1+ctrl", "+") == ("KLF1",)
    assert data.parse_perturbation("MAP2K3+MAP2K6", "+") == ("MAP2K3", "MAP2K6")
    assert data.parse_perturbation("MAP2K6+MAP2K3", "+") == ("MAP2K3", "MAP2K6")


def test_control_labels_map_to_control():
    for label in ["control", "non-targeting", "ctrl+ctrl"]:
        assert data.canonical_name(data.parse_perturbation(label, "+")) == data.CONTROL


def test_no_separator_keeps_hyphenated_gene_names():
    assert data.parse_perturbation("HLA-A", None) == ("HLA-A",)


def test_get_counts_detects_integer_matrix():
    X = sp.csr_matrix(np.array([[0, 2, 1], [3, 0, 0]], dtype=float))
    a = ad.AnnData(X)
    M, ok = data.get_counts(a)
    assert ok and M.shape == (2, 3)


def test_get_counts_rejects_normalised_matrix():
    a = ad.AnnData(np.array([[0.1, 2.5], [1.2, 0.0]], dtype=np.float32))
    _, ok = data.get_counts(a)
    assert not ok


def _raw_dataset(seed=0):
    rng = np.random.default_rng(seed)
    genes = [f"g{i}" for i in range(60)]
    labels = ["non-targeting"] * 120 + ["g0"] * 40 + ["g1"] * 40 + ["g0+g1"] * 40 + ["rare"] * 3
    X = rng.poisson(3, (len(labels), len(genes))).astype(np.float32)
    obs = pd.DataFrame({"pert": labels, "gem": rng.choice(["a", "b"], len(labels))})
    return ad.AnnData(X, obs=obs, var=pd.DataFrame(index=genes))


def test_prepare_standardises_layout():
    adata = _raw_dataset()
    ds = {"name": "raw", "perturbation_col": "pert", "combo_sep": "+", "batch_col": "gem",
          "guide_col": None, "var_symbol_col": None, "max_perturbations": None}
    pp = {"n_top_genes": 30, "min_cells_per_perturbation": 20,
          "max_cells_per_perturbation": 30, "max_control_cells": 50}
    out = data.prepare(adata, ds, pp, seed=0)

    assert set(out.obs["perturbation"]) == {"control", "g0", "g1", "g0+g1"}
    assert out.obs["perturbation"].value_counts()["control"] == 50
    assert out.obs["perturbation"].value_counts()["g0"] == 30
    assert out.uns["has_counts"] is True
    assert out.layers["counts"].shape == out.shape
    assert {"g0", "g1"} <= set(out.var_names)
    assert out.obs.loc[out.obs["perturbation"] == "g0+g1", "n_perts"].eq(2).all()
