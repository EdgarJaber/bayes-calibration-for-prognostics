"""Ensemble Kalman smoothing of surrogate trajectories against field data."""

from __future__ import annotations

import numpy as np


def ensemble_kalman_smoother(Y: np.ndarray, obs_indices, obs_values,
                             obs_std: float) -> np.ndarray:
    """Sequential ensemble Kalman smoother for one observation channel.

    Parameters
    ----------
    Y : np.ndarray, shape (N, T)
        Ensemble of N trajectories discretised on T time points.
    obs_indices : sequence of int
        Columns of ``Y`` (time indices) at which the channel is observed.
    obs_values : sequence of float
        Observations, addressed by the *same* column index: the value observed at
        column ``t`` is ``obs_values[t]`` -- so pass an array of length ``T`` (or at
        least ``max(obs_indices) + 1``) holding the observation at its column.
    obs_std : float
        Observation-noise standard deviation.

    Returns
    -------
    np.ndarray, shape (N, T)
        The smoothed ensemble (``Y`` itself is left untouched).
    """
    Y = Y.copy()
    N, _ = Y.shape
    R = obs_std ** 2

    for t_obs in obs_indices:
        obs = obs_values[t_obs]

        m = np.mean(Y, axis=0)
        A = Y - m                                   # anomalies, (N, T)
        C_ty = (A.T @ A[:, t_obs]) / (N - 1)        # cross-cov, (T,)
        C_yy = np.var(Y[:, t_obs], ddof=1)          # obs variance

        K = C_ty / (C_yy + R)                       # Kalman gain, (T,)
        innovation = obs - Y[:, t_obs]              # (N,)
        Y += np.outer(innovation, K)

    return Y
