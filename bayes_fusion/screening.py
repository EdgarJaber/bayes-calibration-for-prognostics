"""Sensitivity screening (HSIC) and distributional-divergence utilities."""

from __future__ import annotations

import numpy as np
import openturns as ot

from .hsic import HSIC


def select_active_index(X: np.ndarray, y: np.ndarray) -> int:
    """Return the input index with the largest time-averaged R2-HSIC index.

    This is the variable the calibration step will treat as the single active
    (identifiable) parameter for the current cycle.
    """
    hsic = HSIC(ot.Sample(X), ot.Sample(y))
    r2_hsic, _ = hsic.r2_hsic()
    mean_indices = r2_hsic.mean(axis=1)
    return int(np.argmax(mean_indices))


def kl_divergence(a: np.ndarray, b: np.ndarray) -> float:
    """Discrete KL divergence, safe against zeros in ``a`` (0 log 0 := 0)."""
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


def estimate_kl(posterior: ot.Distribution, prior_marginal: ot.Distribution,
                n_samples: int, bins: int) -> float:
    """Histogram-based KL(posterior || prior) on the active 1-D marginal."""
    post = np.asarray(posterior.getSample(n_samples)).reshape(-1)
    pri = np.asarray(prior_marginal.getSample(n_samples)).reshape(-1)

    post_pdf, _ = np.histogram(post, bins=bins, density=True)
    pri_pdf, _ = np.histogram(pri, bins=bins, density=True)

    post_pdf = post_pdf / np.sum(post_pdf)
    pri_pdf = pri_pdf / np.sum(pri_pdf)

    return kl_divergence(post_pdf, pri_pdf)
