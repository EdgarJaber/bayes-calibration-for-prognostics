"""Cycle-by-cycle ensemble Kalman smoothing of the clogging trajectories."""

from __future__ import annotations

import numpy as np

from bayes_fusion.smoothing import ensemble_kalman_smoother


def smooth_cycles(traj: np.ndarray, dataset, cfg, cleaning_time_indices,
                  align_observations: bool = False):
    """Apply the smoother cycle-by-cycle and concatenate along time.

    Returns ``(smoothed, time_axis)`` where ``time_axis`` is the concatenated per-cycle
    grid the smoother actually used -- this grid interleaves the observation timestamps
    with the simulation grid, so it is generally longer than the simulation grid and its
    length depends on the number of observations (hence on ``k``). Callers must slice the
    prognostics window on this returned axis, not on the simulation-grid indices.

    Parameters
    ----------
    align_observations : bool
        ``False`` (default) reproduces the original notebook and the published numbers.
        There, ``ind`` -- the rank of each observation inside its channel -- is used as the
        *column* of the trajectory ensemble, so the i-th observation updates the i-th
        column of the cycle's time axis instead of the column holding its own time, and
        the axis only contains the first channel's observation times.
        ``True`` builds one time axis per cycle from the simulation grid and the
        observation times of *all* channels, and assimilates every observation at the
        column of its own time.
    """
    if align_observations:
        return _smooth_cycles_aligned(traj, dataset, cfg, cleaning_time_indices)

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

            time_indices = _cycle_time_axis(sim_time, cleaning_time_indices, i, L, obs_indices)

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


def _cycle_time_axis(sim_time, cleaning_time_indices, i, L, obs_times):
    """Sorted union of the observation times and the cycle's slice of the simulation grid."""
    if i == 0:
        grid = sim_time[0:int(cleaning_time_indices[i])]
    elif i == L - 1:
        grid = sim_time[int(cleaning_time_indices[i - 1]):]
    else:
        grid = sim_time[int(cleaning_time_indices[i - 1]):int(cleaning_time_indices[i])]
    return np.sort(np.concatenate([obs_times, grid])).astype(int)


def _smooth_cycles_aligned(traj, dataset, cfg, cleaning_time_indices):
    sim_time = dataset.simulation_time
    data_td = dataset.data_with_time_division
    obs_std = cfg.obs_std()
    L, q = len(data_td), len(obs_std)

    y_list, time_list = [], []
    for i in range(L):
        channels = [(data_td[i][j].T[0].astype(int), data_td[i][j].T[1]) for j in range(q)]
        all_times = np.concatenate([t for t, _ in channels])
        time_axis = np.unique(_cycle_time_axis(sim_time, cleaning_time_indices, i, L, all_times))
        y = np.asarray([np.interp(time_axis, sim_time, traj[k, :]) for k in range(len(traj))])

        for j, (times, values) in enumerate(channels):
            for t, v in zip(times, values):
                col = int(np.searchsorted(time_axis, t))
                obs = np.zeros(y.shape[1])
                obs[col] = v
                y = ensemble_kalman_smoother(y, [col], obs, obs_std=obs_std[j])
        y_list.append(y)
        time_list.append(time_axis)

    return np.concatenate(tuple(y_list), axis=1), np.concatenate(tuple(time_list))
