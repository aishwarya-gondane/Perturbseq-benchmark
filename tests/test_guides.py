import numpy as np

from psbench import guides


def test_simulation_shapes_and_truth_rate():
    counts, truth = guides.simulate_guide_counts(n_cells=2000, n_guides=30, moi=2.0, seed=0)
    assert counts.shape == truth.shape == (2000, 30)
    assert abs(truth.sum(1).mean() - 2.0) < 0.2
    assert counts[truth].mean() > counts[~truth].mean()


def test_all_methods_recover_strong_signal():
    counts, truth = guides.simulate_guide_counts(
        n_cells=2000, n_guides=30, moi=1.0, signal_mean=300, ambient_mean=2, seed=0
    )
    for name in ["threshold", "ratio", "mixture"]:
        pred = guides.METHODS[name](counts)
        assert guides.evaluate_assignment(pred, truth)["f1"] > 0.9, name


def test_top_guide_cannot_call_multiplets():
    counts, truth = guides.simulate_guide_counts(
        n_cells=2000, n_guides=30, moi=5.0, signal_mean=300, seed=0
    )
    pred = guides.assign_top(counts)
    assert pred.sum(1).max() <= 1
    assert guides.evaluate_assignment(pred, truth)["recall"] < 0.5


def test_evaluate_assignment_perfect_and_empty():
    truth = np.eye(5, dtype=bool)
    assert guides.evaluate_assignment(truth, truth)["f1"] == 1.0
    assert guides.evaluate_assignment(np.zeros_like(truth), truth)["f1"] == 0.0


def test_summarize_calls_sums_to_one():
    pred = np.array([[0, 0], [1, 0], [1, 1]], dtype=bool)
    s = guides.summarize_calls(pred)
    assert abs(sum(s.values()) - 1.0) < 1e-12


def test_benchmark_table():
    cfg = {"n_cells": 500, "n_guides": 10, "moi": [1.0], "signal_mean": [100],
           "ambient_mean": 3.0}
    df = guides.benchmark(cfg)
    assert set(df["method"]) == set(guides.METHODS)
    assert df["f1"].between(0, 1).all()
