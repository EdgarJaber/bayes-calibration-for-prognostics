"""Configuration of the steam-generator (SG) clogging application.

The generic settings come from :mod:`bayes_fusion.config`; this module adds the
THYC-Puffer-DEPO clogging defaults (prior ranges, MCMC step sizes) and the
application-level fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from bayes_fusion.config import (AssimilationConfig, FusionConfig, MCMCConfig,
                                 MetamodelConfig, PriorSpec)

__all__ = ["PipelineConfig", "PriorSpec", "MCMCConfig", "AssimilationConfig",
           "MetamodelConfig", "clogging_prior", "clogging_step_sizes"]


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
    clogging_threshold: float | None = None    # tau_c^* (% of clogging) defining failure
    obs_std_percent: Sequence[float] = field(default_factory=lambda: [8, 5])
    obs_std_ci: float = 1.96            # divisor converting a 95% CI to a std

    def obs_std(self) -> list[float]:
        return [s / self.obs_std_ci for s in self.obs_std_percent]
