"""Configuration of the steam-generator (SG) clogging application.

The generic settings come from :mod:`bayes_fusion.config`; this module adds the
THYC-Puffer-DEPO clogging defaults (prior ranges, MCMC step sizes) and the
application-level fields.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Sequence

from bayes_fusion.config import (AssimilationConfig, FusionConfig, MCMCConfig,
                                 MetamodelConfig, PriorSpec)

__all__ = ["PipelineConfig", "PriorSpec", "MCMCConfig", "AssimilationConfig",
           "MetamodelConfig", "clogging_prior", "clogging_step_sizes",
           "load_thresholds", "threshold_symbol"]


def load_thresholds(base_directory: str) -> dict[str, float]:
    """Failure thresholds tau_c^{*,A} and tau_c^{*,B}, read from ``<base_directory>/thresholds.json``.

    The file ``{"SGA": <tau_c^{*,A}>, "SGB": <tau_c^{*,B}>}`` is part of the confidential data
    (git-ignored): the values are deliberately not stored in the code or the configurations.
    """
    path = os.path.join(base_directory, "thresholds.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} is missing: it must hold the failure thresholds of the steam generators, "
            'as {"SGA": ..., "SGB": ...} (it is distributed with the data, see DATA/README.md).')
    with open(path) as fh:
        return {k: float(v) for k, v in json.load(fh).items()}


def threshold_symbol(sg_tag: str) -> str:
    """LaTeX symbol of a steam generator's threshold: ``SGA`` -> ``\\tau_c^{\\ast,\\mathrm{A}}``."""
    return rf"\tau_c^{{\ast,\mathrm{{{sg_tag[-1]}}}}}"


def clogging_prior() -> PriorSpec:
    """The seven THYC / DEPOTHYC clogging parameters and their Uniform ranges."""
    return PriorSpec(
        bounds=[
            (100.0, 103.0),     # alpha
            (0.02, 0.025),      # beta
            (0.2, 0.5),         # eps_e
            (0.01, 0.3),        # eps_c
            (0.5e-6, 10.0e-6),  # d_p
            (1.0e-9, 8.0e-9),   # g_p0
            (0.0, 15e-4),       # a_v
        ],
        names=["alpha", "beta", "eps_e", "eps_c", "d_p", "g_p0", "a_v"],
    )


def clogging_step_sizes() -> list[float]:
    return [10, 0.01, 0.01, 0.05, 5e-6, 1.5e-9, 1e-3]


def _clogging_mcmc() -> MCMCConfig:
    return MCMCConfig(step_sizes=clogging_step_sizes())


@dataclass(kw_only=True)
class PipelineConfig(FusionConfig):
    """Everything needed to run the clogging prognostics pipeline for one SG."""

    base_directory: str
    input_dimension: int = 7
    prior: PriorSpec = field(default_factory=clogging_prior)
    mcmc: MCMCConfig = field(default_factory=_clogging_mcmc)

    n_cycles: int = 2                   # number of CC/PC assimilation scenarios

    # Prognostics / RUL
    clogging_threshold: float | None = None   # tau_c^{*} of the SG, see load_thresholds()
    obs_std_percent: Sequence[float] = field(default_factory=lambda: [8, 5])
    obs_std_ci: float = 1.96            # divisor converting a 95% CI to a std

    def obs_std(self) -> list[float]:
        return [s / self.obs_std_ci for s in self.obs_std_percent]
