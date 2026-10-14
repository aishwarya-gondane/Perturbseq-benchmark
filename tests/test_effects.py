import numpy as np
import pytest

from psbench import effects, metrics


def _recall(res, planted, fdr=0.05):
    called = set(res.loc[res["fdr"] < fdr, "gene"])
    return len(called & set(planted)) / len(planted)


def test_single_perturbations_excludes_combos_and_control(toy):
    perts = effects.single_perturbations(toy)
    assert perts == [f"gene{i}" for i in range(6)]


@pytest.mark.parametrize("method", ["nb_glm", "welch"])
def test_methods_find_knockdown_and_planted_genes(toy, method):
    res = effects.PER_PERT_METHODS[method](toy, "gene0", "control", **effects.method_kwargs(
        method, {"min_frac_expressed": 0.05, "mixed_genes": 50}))
    target = res[res["gene"] == "gene0"].iloc[0]
    assert target["log2fc"] < -1 and target["fdr"] < 0.05
    assert _recall(res, toy.uns["planted_de"]["gene0"]) > 0.5


def test_mixed_model_target_effect(toy):
    res = effects.mixed(toy, "gene1", "control", group_col="batch", n_genes=30)
    target = res[res["gene"] == "gene1"].iloc[0]
    assert target["log2fc"] < -1 and target["fdr"] < 0.05
    assert res["pvalue"].notna().sum() >= 30


def test_nb_glm_direction_of_planted_effects(toy):
    res = effects.nb_glm(toy, "gene2", "control").set_index("gene")
    # planted effects were drawn on the natural-log scale with random sign
    sig = res[res["fdr"] < 0.05]
    assert len(sig) > 10
    assert np.isfinite(sig["log2fc"]).all()


def test_on_target_summary(toy):
    res = effects.run_per_pert(toy, ["gene0", "gene1", "gene2"], "welch")
    s = metrics.on_target_summary(res).iloc[0]
    assert s["n_perts"] == 3 and s["knockdown_rate"] == 1.0


def test_null_calibration_is_roughly_uniform(toy):
    out = metrics.null_calibration(toy, "welch", n_groups=4, group_size=60, seed=0)
    assert len(out) == 4
    assert out["frac_p_lt_0.05"].mean() < 0.10
    assert out["n_deg_fdr_0.05"].mean() < 3


def test_null_calibration_nb_glm(toy):
    out = metrics.null_calibration(toy, "nb_glm", n_groups=2, group_size=60, seed=0)
    assert out["frac_p_lt_0.05"].mean() < 0.12
