"""Cycle-by-cycle ensemble Kalman smoothing of the clogging trajectories."""

from __future__ import annotations

import numpy as np

from bayes_fusion.smoothing import ensemble_kalman_smoother


def smooth_cycles(traj: np.ndarray, dataset, cfg, cleaning_time_indices) -> np.ndarray:
    """Apply the smoother cycle-by-cycle and concatenate along time.

    Reproduces the notebook's nested-loop smoothing over ``L`` cycles and ``q``
    observation channels. Returns ``(smoothed, time_axis)`` where ``time_axis``
    is the concatenated per-cycle grid the smoother actually used -- this grid
    interleaves the observation timestamps with the simulation grid, so it is
    generally longer than ``sim_time`` and its length depends on the number of
    observations (hence on ``k``). Callers must slice the prognostics window on
    this returned axis, not on the simulation-grid indices.

    Known issue (inherited from the original notebook, kept so that published numbers stay
    reproducible): ``ind`` below is the rank of each observation inside its channel, and it is
    used as the *column* of the trajectory ensemble, so the i-th observation updates the i-th
    column of the cycle's time axis rather than the column holding its own time.
    """
    sim_time = dataset.simulation_time
    data_td = dataset.data_with_time_division
    obs_std = cfg.obs_std()

    L = len(data_td)
    q = len(obs_std)

    y_list = []
    time_list = []
    for i in range(L):
        y = None
        cycle_time = None
        for j in range(q):
            obs_indices = data_td[i][j].T[0].astype(int)
            obs_values = data_td[i][j].T[1]

            if i == 0:
                time_indices = np.sort(np.concatenate([
                    obs_indices, sim_time[0:int(cleaning_time_indices[i])]
                ])).astype(int)
            elif i == L - 1:
                time_indices = np.sort(np.concatenate([
                    obs_indices, sim_time[int(cleaning_time_indices[i - 1]):]
                ])).astype(int)
            else:
                time_indices = np.sort(np.concatenate([
                    obs_indices,
                    sim_time[int(cleaning_time_indices[i - 1]):int(cleaning_time_indices[i])]
                ])).astype(int)

            ind = [
                int(np.where(obs_indices == t)[0][0])
                for t in time_indices if t in obs_indices
            ]

            if j == 0:
                y = np.asarray([
                    np.interp(time_indices, sim_time, traj[k, :])
                    for k in range(len(traj))
                ])
                cycle_time = time_indices

            y = ensemble_kalman_smoother(y, ind, obs_values, obs_std=obs_std[j])
        y_list.append(y)
        time_list.append(cycle_time)

    smoothed = np.concatenate(tuple(y_list), axis=1)
    time_axis = np.concatenate(tuple(time_list))
    return smoothed, time_axis