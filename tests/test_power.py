import numpy as np

from psbench import power


def test_estimated_params_are_valid(toy):
    mu, alpha = power.estimate_params(toy)
    assert len(mu) == len(alpha) > 50
    assert (mu > 0).all() and (alpha > 0).all()


def test_power_increases_with_cells_and_effect_size(toy):
    mu, alpha = power.estimate_params(toy)
    df = power.simulate_power(mu, alpha, [20, 200], [0.25, 1.0], n_control=300, n_rep=3, seed=0)
    m = df.groupby(["n_cells", "log2fc"])["power"].mean()
    assert m[(200, 1.0)] > m[(20, 1.0)]
    assert m[(200, 1.0)] > m[(200, 0.25)]
    assert m[(200, 1.0)] > 0.5


def test_observed_fdr_is_controlled_for_large_effects(toy):
    mu, alpha = power.estimate_params(toy)
    df = power.simulate_power(mu, alpha, [200], [1.0], n_control=300, n_rep=5, seed=1)
    assert df["observed_fdr"].mean() < 0.15


def test_zero_effect_has_no_power_to_speak_of(toy):
    mu, alpha = power.estimate_params(toy)
    df = power.simulate_power(mu, alpha, [100], [0.0], n_control=300, n_rep=3, seed=2)
    assert np.isfinite(df["power"]).all()
    assert df["power"].mean() < 0.1


def test_downsample_recovery_grows_with_fraction(toy):
    df = power.downsample_recovery(toy, [0.1, 1.0], n_rep=2, max_perts=2)
    m = df.groupby("fraction")["recovery"].mean()
    assert m[1.0] >= m[0.1]
    assert m[1.0] > 0.9
