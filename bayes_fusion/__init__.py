"""Bayesian fusion of heterogeneous data for degradation prognostics.

Application-independent building blocks of the methodology of

    Jaber, Remy, Chabridon, Mougeot, Lucor -- *Fusion of heterogeneous data for robust
    degradation prognostics*, Reliability Engineering & System Safety (2026).

* :mod:`bayes_fusion.metamodels` -- scalar GP, vector PCE and Karhunen-Loeve surrogates
* :mod:`bayes_fusion.hsic` -- HSIC sensitivity indices
* :mod:`bayes_fusion.screening` -- active-variable selection and KL divergence
* :mod:`bayes_fusion.mcmc` -- RWMH-MCMC with the aggregated (multi-surrogate) likelihood
* :mod:`bayes_fusion.assimilation` -- the refine-until-converged calibration loop
* :mod:`bayes_fusion.smoothing` -- ensemble Kalman smoother
* :mod:`bayes_fusion.prognostics` -- remaining-useful-life utilities
"""

from .config import (AssimilationConfig, FusionConfig, MCMCConfig, MetamodelConfig,
                     PriorSpec)
from .hsic import HSIC
from .mcmc import BayesCalibrationMCMC, sobol_simplex_weights
from .prognostics import (bootstrap_align, compute_prognostics_trajectories,
                          first_crossing_rul)
from .restriction import RestrictedModel
from .screening import estimate_kl, kl_divergence, select_active_index
from .smoothing import ensemble_kalman_smoother

__all__ = [
    "PriorSpec", "MCMCConfig", "AssimilationConfig", "MetamodelConfig", "FusionConfig",
    "HSIC", "BayesCalibrationMCMC", "sobol_simplex_weights",
    "select_active_index", "kl_divergence", "estimate_kl",
    "ensemble_kalman_smoother", "RestrictedModel",
    "first_crossing_rul", "compute_prognostics_trajectories", "bootstrap_align",
]
