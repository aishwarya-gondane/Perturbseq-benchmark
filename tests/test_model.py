import numpy as np
import torch

from psbench import model

CFG = {"folds": 3, "emb": 8, "hidden": 32, "dropout": 0.0, "epochs": 40,
       "lr": 0.01, "weight_decay": 0.0, "patience": 10, "min_cells": 10}


def test_table_contains_singles_and_doubles(toy):
    t = model.build_table(toy)
    assert len(t.singles) == 6 and len(t.doubles) == 15
    assert all(v.shape == (toy.n_vars,) for v in t.singles.values())


def test_additive_baseline_is_sum_of_singles(toy):
    t = model.build_table(toy)
    np.testing.assert_allclose(
        model.additive(t, "gene0+gene1"), t.singles["gene0"] + t.singles["gene1"]
    )


def test_pair_model_is_symmetric_and_padding_is_zero():
    m = model.PairMLP(n_embed=5, n_genes=7, emb=4, hidden=8, dropout=0.0).eval()
    a, b = torch.tensor([[1, 2]]), torch.tensor([[2, 1]])
    torch.testing.assert_close(m(a), m(b))
    assert torch.all(m.emb.weight[0] == 0)
    assert m(torch.tensor([[3, 0]])).shape == (1, 7)


def test_score_perfect_prediction():
    truth = np.random.default_rng(0).normal(size=100)
    s = model.score(truth, truth)
    assert s["mse"] == 0 and abs(s["pearson"] - 1) < 1e-9


def test_score_constant_prediction_has_zero_correlation():
    truth = np.random.default_rng(0).normal(size=100)
    assert model.score(np.zeros(100), truth)["pearson"] == 0.0


def test_cross_validation_runs_and_covers_every_double(toy):
    cv = model.run_cv(toy, CFG, seed=0)
    assert set(cv["model"]) == {"additive", "control", "mean_double", "mlp_direct", "mlp_residual"}
    per_model = cv.groupby("model")["condition"].nunique()
    assert (per_model == 15).all()
    assert np.isfinite(cv[["mse", "pearson", "mse_top", "pearson_top"]].to_numpy()).all()


def test_additive_beats_control_when_effects_are_mostly_additive(toy):
    cv = model.run_cv(toy, CFG, seed=0)
    mse = cv.groupby("model")["mse"].mean()
    assert mse["additive"] < mse["control"]
