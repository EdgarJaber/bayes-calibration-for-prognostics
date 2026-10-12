"""Remaining-useful-life (RUL) computation from emulated trajectories."""

from __future__ import annotations

import numpy as np
import openturns as ot
from sklearn.utils import resample


def first_crossing_rul(trajectories: np.ndarray, pc_time: np.ndarray,
                       threshold: float) -> np.ndarray:
    """Time from the start of the prognostics window to first threshold crossing.

    For each trajectory, find the first time index where the trajectory exceeds
    ``threshold`` and record the elapsed time since ``pc_time[0]``. Trajectories
    that never cross are omitted (matching the notebook behaviour).
    """
    ruls = []
    for i in range(trajectories.shape[0]):
        crossings = np.where(trajectories[i] > threshold)[0]
        if crossings.size:
            ruls.append(pc_time[crossings[0]] - pc_time[0])
    return np.asarray(ruls)


def compute_prognostics_trajectories(base_surrogate, global_scaler,
                                     prior_sample, posterior_sample,
                                     smoothed_trajectories,
                                     simulation_time, pc_time, split_index):
    """Assemble the three trajectory ensembles compared in the RUL study.

    Returns ``(prior, bmu, bmu_enks)`` trajectories, each interpolated onto the
    prognostics-window time grid ``pc_time`` where applicable.
    """
    prior_traj = base_surrogate.predict(global_scaler.transform(prior_sample))
    prior_traj = np.asarray([
        np.interp(pc_time, simulation_time, prior_traj[i, :])
        for i in range(len(prior_traj))
    ])

    bmu_traj = base_surrogate.predict(global_scaler.transform(posterior_sample))
    bmu_traj = np.asarray([
        np.interp(pc_time, simulation_time, bmu_traj[i, :])
        for i in range(len(bmu_traj))
    ])

    bmu_enks_traj = smoothed_trajectories[:, split_index + 2:]

    return prior_traj, bmu_traj, bmu_enks_traj


def bootstrap_align(*rul_arrays, random_state: int = 42):
    """Bootstrap-resample every RUL array to a common size for comparison."""
    target = max(len(a) for a in rul_arrays)
    return tuple(
        resample(a, n_samples=target, random_state=random_state)
        for a in rul_arrays
    )
