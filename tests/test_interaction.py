import numpy as np

from psbench import interaction


def test_combos_found(toy):
    found = interaction.combos(toy)
    assert len(found) == 15
    assert ("gene0", "gene1", "gene0+gene1") in found


def test_planted_interaction_genes_are_detected(toy):
    genes, summary = interaction.interaction_for_combo(toy, "gene0", "gene1", "gene0+gene1")
    planted = set(toy.uns["planted_interaction"]["gene0+gene1"])
    called = set(genes.loc[genes["fdr"] < 0.05, "gene"])
    assert len(planted & called) / len(planted) > 0.5
    # genes with no planted interaction should rarely be called
    false_calls = called - planted
    assert len(false_calls) < 0.1 * toy.n_vars


def test_summary_fields(toy):
    _, s = interaction.interaction_for_combo(toy, "gene0", "gene1", "gene0+gene1")
    assert s["n_cells_ab"] == 120
    assert np.isfinite([s["c1"], s["c2"], s["r2_dosage"], s["additive_error"]]).all()
    assert s["magnitude_ratio"] > 0


def test_no_interaction_gives_low_additive_error():
    from psbench.simulate import toy_dataset

    clean = toy_dataset(n_interaction=0, seed=3)
    summaries, _ = interaction.run_interactions(clean)
    noisy = toy_dataset(n_interaction=30, seed=3)
    noisy_summaries, _ = interaction.run_interactions(noisy)
    assert summaries["additive_error"].mean() < noisy_summaries["additive_error"].mean()


def test_too_few_cells_returns_none(toy):
    assert interaction.interaction_for_combo(
        toy, "gene0", "gene1", "gene0+gene1", min_cells=10_000
    ) is None


def test_run_interactions_table(toy):
    summary, genes = interaction.run_interactions(toy)
    assert len(summary) == 15
    assert set(summary["class"]) <= {"synergistic", "buffering", "additive"}
    assert (genes["fdr"] < 0.05).all()
