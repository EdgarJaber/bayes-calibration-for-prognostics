"""Tests of the cycle-wise EnKS (no external data needed). Run from the application root."""

import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from clogging_prognostics import smooth_cycles


def _problem():
    rng = np.random.default_rng(0)
    sim = np.arange(100, 600, 10).astype(float)              # 50 grid points
    traj = rng.normal(size=(200, sim.size)).cumsum(axis=1)
    obs_t, obs_v = np.array([300.0, 450.0]), np.array([20.0, -15.0])
    ds = types.SimpleNamespace(simulation_time=sim,
                               data_with_time_division=[[np.column_stack([obs_t, obs_v]),
                                                         np.zeros((0, 2))]])
    cfg = types.SimpleNamespace(obs_std=lambda: [0.5, 0.5])
    return traj, ds, cfg, obs_t, obs_v


def test_smoothing_pulls_the_ensemble_to_the_observations_at_their_times():
    traj, ds, cfg, obs_t, obs_v = _problem()
    sm, t = smooth_cycles(traj, ds, cfg, [50])
    assert np.all(np.diff(t) > 0)
    for tt, v in zip(obs_t, obs_v):
        c = int(np.searchsorted(t, tt))
        assert abs(sm[:, c].mean() - v) < 1.0                  # pulled onto the observation
        assert sm[:, c].std() < 1.0                            # and its spread collapsed


def test_time_axis_contains_the_observation_times_of_every_channel():
    traj, ds, cfg, _, _ = _problem()
    ds.data_with_time_division[0][1] = np.array([[205.0, 3.0]])   # a second channel, off the grid
    sm, t = smooth_cycles(traj, ds, cfg, [50])
    assert 205 in t and 300 in t and 450 in t
    assert sm.shape == (traj.shape[0], t.size)
    c = int(np.searchsorted(t, 205))
    assert abs(sm[:, c].mean() - 3.0) < 1.0
