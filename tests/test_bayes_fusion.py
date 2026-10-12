"""Unit tests of the application-independent core (no external data needed).

Run with ``pytest tests`` (or call the ``test_*`` functions directly).
"""

import numpy as np
import openturns as ot

from bayes_fusion import (BayesCalibrationMCMC, FusionConfig, PriorSpec, RestrictedModel,
                          ensemble_kalman_smoother, first_crossing_rul, sobol_simplex_weights)


def test_sobol_weights_lie_on_the_simplex_and_are_deterministic():
    w = sobol_simplex_weights(7, 256, seed=3)
    assert w.shape == (256, 7)
    assert np.all(w >= 0) and np.allclose(w.sum(axis=1), 1.0)
    assert np.array_equal(w, sobol_simplex_weights(7, 256, seed=3))
    assert not np.array_equal(w, sobol_simplex_weights(7, 256, seed=4))


class _Linear:
    """Surrogate returning slope * x * time-grid, as the calibration function expects."""

    def __init__(self, slope, n_t=12):
        self.slope, self.t = slope, np.linspace(1.0, 2.0, n_t)

    def predict(self, x):
        return (self.slope * float(np.asarray(x).reshape(-1)[0]) * self.t).reshape(1, -1)


def _problem(n_models=4):
    rng = np.random.default_rng(0)
    models = [_Linear(s) for s in rng.uniform(0.9, 1.1, n_models)]
    t = np.linspace(1.0, 2.0, 12)
    data = [np.column_stack([np.arange(3), 1.5 * t[[1, 4, 9]] + 0.01]),
            np.column_stack([np.arange(4), 1.5 * t[[0, 5, 7, 11]] - 0.01])]
    idx = [np.array([1, 4, 9]), np.array([0, 5, 7, 11])]
    # positions in the sorted timeline of the union of the observed columns
    union = np.unique(np.concatenate(idx))
    return models, data, [np.searchsorted(union, i) for i in idx], union


def _calibration_functions(models, union):
    class _Sub:
        def __init__(self, m): self.m = m
        def predict(self, x): return self.m.predict(x)[:, union]
    return [BayesCalibrationMCMC.CalibrationFunction(_Sub(m), None, [np.arange(len(union))], None)
            for m in models]


def test_aggregated_likelihood_is_deterministic_and_equals_logsumexp():
    models, data, tidx, union = _problem()
    cals = _calibration_functions(models, union)
    lik = BayesCalibrationMCMC.LikelihoodFunction(data, tidx, cals, nb_weights=128)()
    a, b = lik([1.0])[0], lik([1.0])[0]
    assert a == b                                             # no resampling noise

    w = sobol_simplex_weights(len(models), 128, seed=0)
    out = w @ np.stack([c([1.0]).reshape(-1) for c in cals])   # (128, n_union)
    ll = np.zeros(128)
    for d, i in zip(data, tidx):
        ll += -np.log(np.sum((d[:, 1] - out[:, i]) ** 2, axis=1)) * len(d) / 2.0
    ref = np.log(np.mean(np.exp(ll - ll.max()))) + ll.max()
    assert np.isclose(a, ref)


def test_aggregated_likelihood_ignores_an_empty_channel():
    models, data, tidx, union = _problem()
    cals = _calibration_functions(models, union)
    empty = np.zeros((0, 2))
    lik = BayesCalibrationMCMC.LikelihoodFunction(data + [empty], tidx + [np.array([], dtype=int)],
                                                  cals, nb_weights=64)()
    assert np.isfinite(lik([1.0])[0])


def test_restricted_model_sweeps_only_the_active_input():
    f = lambda X: np.column_stack([X[:, 0] + 10 * X[:, 1], X[:, 2]])
    r = RestrictedModel(f, index=1, ref_point=[1.0, 0.0, 5.0])
    assert np.allclose(r.predict([2.0, 3.0]), [[21.0, 5.0], [31.0, 5.0]])
    g = RestrictedModel(ot.SymbolicFunction(['x', 'y'], ['x*y']), index=0, ref_point=[0.0, 4.0])
    assert np.allclose(g.predict([3.0]), [[12.0]])


def test_first_crossing_rul_is_time_since_window_start_and_skips_non_crossing():
    t = np.array([10.0, 20.0, 30.0, 40.0])
    traj = np.array([[0, 1, 5, 6], [0, 0, 0, 0], [9, 9, 9, 9]], dtype=float)
    assert np.allclose(first_crossing_rul(traj, t, 4.0), [20.0, 0.0])


def test_enks_pulls_the_ensemble_towards_the_observation_and_keeps_the_input():
    rng = np.random.default_rng(1)
    Y = rng.normal(0.0, 1.0, (200, 6)).cumsum(axis=1)
    Y0 = Y.copy()
    obs = np.zeros(6)
    obs[3] = 5.0
    S = ensemble_kalman_smoother(Y, [3], obs, obs_std=0.1)
    assert np.array_equal(Y, Y0)
    assert abs(S[:, 3].mean() - 5.0) < abs(Y0[:, 3].mean() - 5.0)
    assert S[:, 3].std() < Y0[:, 3].std()


def test_fusion_config_requires_a_prior_and_dimension():
    prior = PriorSpec(bounds=[(0, 1), (2, 3)], names=['a', 'b'])
    cfg = FusionConfig(input_dimension=2, prior=prior)
    assert cfg.prior.dimension == 2 and cfg.mcmc.nb_weights == 1024 and cfg.train_size == 0.75
    assert len(cfg.prior.marginals()) == 2
