"""Configuration objects of the Bayesian data-fusion methodology.

Everything that is application independent -- prior, MCMC, assimilation loop and
surrogate settings -- is gathered in typed dataclasses. An application subclasses
:class:`FusionConfig` to add its own fields (see ``applications/sg_clogging``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import openturns as ot


@dataclass
class PriorSpec:
    """Independent marginal Uniform priors on the physical input parameters."""

    bounds: Sequence[tuple[float, float]]
    names: Sequence[str]

    def marginals(self) -> list[ot.Distribution]:
        """Return a *fresh* list of Uniform marginals (never share state)."""
        return [ot.Uniform(lo, hi) for lo, hi in self.bounds]

    def composed(self) -> ot.JointDistribution:
        return ot.JointDistribution(self.marginals())

    @property
    def dimension(self) -> int:
        return len(self.bounds)


@dataclass
class MCMCConfig:
    """Random-walk Metropolis-Hastings settings.

    ``nb_weights`` is the size of the fixed scrambled-Sobol design on the simplex used to
    aggregate the calibration surrogates inside the likelihood (a power of two is best).
    ``step_sizes`` holds one half-width of the uniform proposal per input parameter.
    """

    step_sizes: Sequence[float] = ()
    burn_in: int = 100
    n_chains: int = 3
    sample_size: int = 1000
    nb_weights: int = 1024

    def proposals(self) -> list[ot.Distribution]:
        return [ot.Uniform(-s, s) for s in self.step_sizes]


@dataclass
class AssimilationConfig:
    """Iterative prior-refinement loop settings."""

    kl_threshold: float = 1e-7          # stopping tol on KL(posterior || prior)
    kl_bins: int = 100                  # histogram bins for the KL estimate
    n_kl_samples: int = 10000           # samples to estimate the KL divergence
    max_iterations: int = 50            # hard cap to avoid infinite loops
    random_seed: int = 42


@dataclass
class MetamodelConfig:
    """Surrogates: the (trend x kernel) calibration ensemble and the base KLE surrogate."""

    trends: Sequence[str] = field(default_factory=lambda: ["Constant", "Linear", "Quad"])
    kernels: Sequence[str] = field(
        default_factory=lambda: ["AbsExp", "SqExp", "M-1/2", "M-3/2", "M-5/2"]
    )
    base_trend: str = "Constant"        # trend/kernel of the *full* KLE surrogate
    base_kernel: str = "AbsExp"
    kl_threshold: float = 1e-7

    explained_variance_threshold: float = 0.9999  # KLE mode-selection energy target
    nugget: float | None = None          # optional NuggetFactor for GP conditioning


@dataclass(kw_only=True)
class FusionConfig:
    """Application-independent settings of one fusion run."""

    input_dimension: int
    prior: PriorSpec
    n_doe_samples: int = 1000
    train_size: float = 0.75

    mcmc: MCMCConfig = field(default_factory=MCMCConfig)
    assimilation: AssimilationConfig = field(default_factory=AssimilationConfig)
    metamodel: MetamodelConfig = field(default_factory=MetamodelConfig)
