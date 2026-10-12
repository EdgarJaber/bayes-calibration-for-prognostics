"""Iterative Bayesian assimilation for a single clogging cycle (scenario).

One call to :func:`assimilate_cycle` runs the full
refine-until-converged procedure for one CC/PC cycle:

    1. interpolate the DoE outputs onto the field-data time grid;
    2. screen inputs with HSIC and pick the active variable;
    3. build the (trend x kernel) calibration ensemble on that variable;
    4. run MCMC to get the posterior of the active marginal;
    5. estimate KL(posterior || prior);
       * if converged -> stop, return the current marginals;
       * else -> replace the active marginal by the posterior, regenerate the
         DoE from the updated prior via the base surrogate, and repeat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import openturns as ot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .mcmc import BayesCalibrationMCMC
from .screening import select_active_index, estimate_kl
from .surrogates import build_calibration_ensemble


@dataclass
class CycleResult:
    """Outcome of assimilating one clogging cycle."""

    cycle_index: int
    marginals: list                      # refined ot marginals for this cycle
    active_index: int
    n_iterations: int
    converged: bool
    kl_history: list[float]


def _refit_scalers(X: np.ndarray) -> list[StandardScaler]:
    scalers = []
    for k in range(X.shape[1]):
        s = StandardScaler()
        s.fit(X[:, k].reshape(-1, 1))
        scalers.append(s)
    return scalers


def assimilate_cycle(
    cycle_index: int,
    dataset,
    cfg,
    base_surrogate,
    global_scaler,
    per_variable_scalers: list[StandardScaler],
    X0: np.ndarray,
    y0: np.ndarray,
    time_scale: float = 1.0,
) -> CycleResult:
    """Run the refine-until-converged assimilation for one cycle.

    Parameters
    ----------
    cycle_index : int
        Index of the CC/PC cycle to calibrate against.
    dataset : object
        Application data container. Attributes read here: ``simulation_time`` (model
        time grid), ``data_without_time_division`` ((n_obs, 2) array, column 0 = time),
        ``data_with_time_division`` (per cycle: list of (m, 2) channel arrays),
        ``time_indices`` (per cycle: positions of each channel in the sorted timeline)
        and ``n_max`` (size of the Monte-Carlo sample used to regenerate the DoE).
    cfg : PipelineConfig
    base_surrogate : metamodels.KarhunenLoeveMetamodel
        The full surrogate, used to regenerate the DoE from an updated prior.
    global_scaler : StandardScaler
        The single scaler mapping physical inputs -> base-surrogate inputs.
    per_variable_scalers : list[StandardScaler]
        One scaler per input variable (see ``data_io.build_per_variable_scalers``).
        Re-fitted whenever the DoE is regenerated from an updated prior.
    X0, y0 : np.ndarray
        Fresh (scaled X, raw y) DoE for this cycle.
    time_scale : float
        Factor applied to the time axes before the calibration surrogates are built
        (their Karhunen-Loeve mesh lives on the scaled axis). 1 keeps the dataset's own
        unit; the clogging application uses 24 (days -> hours).
    """
    ac = cfg.assimilation
    proposals = cfg.mcmc.proposals()

    marginals = cfg.prior.marginals()          # fresh, never shared across cycles
    scalers = list(per_variable_scalers)       # local copy, updated on refinement
    X, y = X0.copy(), y0.copy()

    field_times = dataset.data_without_time_division[:, 0] * time_scale

    kl_history: list[float] = []
    converged = False

    for iteration in range(ac.max_iterations):
        # 1. interpolate DoE outputs onto the field-data time grid
        y_interp = np.asarray([
            np.interp(field_times, dataset.simulation_time * time_scale, y[i, :])
            for i in range(len(y))
        ])
        X_train, _, y_train_int, _ = train_test_split(
            X, y_interp, train_size=cfg.train_size, random_state=42,
        )

        # 2. HSIC screening
        active_index = select_active_index(X, y)

        # 3. calibration ensemble on the active variable
        metamodels = build_calibration_ensemble(
            X_train, y_train_int, cfg, field_times, active_index,
        )
        # one marginal scaler (the active variable's) per ensemble member
        all_scalers = [scalers[active_index]] * len(metamodels)

        # 4. MCMC posterior on the active marginal
        bayes = BayesCalibrationMCMC(
            data=dataset.data_with_time_division[cycle_index],
            data_time_indices=dataset.time_indices[cycle_index],
            metamodels=metamodels,
            scalers=all_scalers,
        )
        posterior = bayes.mcmc_run(
            support_min=[marginals[active_index].getRange().getLowerBound()[0]],
            support_max=[marginals[active_index].getRange().getUpperBound()[0]],
            proposal=proposals[active_index],
            burning=cfg.mcmc.burn_in,
            n_chains=cfg.mcmc.n_chains,
            sample_size=cfg.mcmc.sample_size,
            nb_weights=cfg.mcmc.nb_weights,
        )

        # 5. convergence via KL(posterior || prior)
        kl = estimate_kl(
            posterior, marginals[active_index],
            n_samples=ac.n_kl_samples, bins=ac.kl_bins,
        )
        kl_history.append(kl)

        if kl < ac.kl_threshold or kl == np.inf:
            converged = True
            break

        # otherwise: update the active marginal and regenerate the DoE
        marginals[active_index] = posterior
        prior = ot.JointDistribution(marginals)
        X_phys = np.asarray(prior.getSample(dataset.n_max))
        scalers = _refit_scalers(X_phys)          # re-fit marginal scalers
        X = global_scaler.transform(X_phys)
        y = base_surrogate.predict(X)

    return CycleResult(
        cycle_index=cycle_index,
        marginals=marginals,
        active_index=active_index,
        n_iterations=iteration + 1,
        converged=converged,
        kl_history=kl_history,
    )
