"""Observation-size sweep experiment on the last clogging scenario.

For a given number ``k`` of field-data observations (the first ``k`` points of
the *last* cycle's field data), this module runs the full pipeline and returns
RUL statistics for the three prognostics stages: prior, BMU (Bayesian model
updating), and BMU+EnKS (after ensemble Kalman smoothing).

Kept in the package (rather than inline in the cluster script) so the sweep
logic is importable and unit-testable, and the SLURM worker stays a thin CLI.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import openturns as ot

from .assimilation import assimilate_cycle
from .data_io import build_doe
from bayes_fusion.prognostics import first_crossing_rul
from .smoothing import smooth_cycles


# --------------------------------------------------------------------------- #
# Statistics container
# --------------------------------------------------------------------------- #
@dataclass
class RULStats:
    """Summary statistics of one RUL sample."""

    stage: str
    n_obs: int
    count: int                 # number of trajectories that crossed the threshold
    n_total: int               # ensemble size (crossing rate = count / n_total)
    mean: float
    std: float
    cv: float                  # coefficient of variation, std / mean
    q05: float
    q25: float
    median: float
    q75: float
    q95: float
    min: float
    max: float

    @classmethod
    def from_sample(cls, stage: str, n_obs: int, rul: np.ndarray,
                    n_total: int) -> "RULStats":
        rul = np.asarray(rul, dtype=float)
        if rul.size == 0:
            nan = float("nan")
            return cls(stage, n_obs, 0, n_total,
                       nan, nan, nan, nan, nan, nan, nan, nan, nan, nan)
        mean = float(np.mean(rul))
        std = float(np.std(rul, ddof=1)) if rul.size > 1 else 0.0
        return cls(
            stage=stage, n_obs=n_obs, count=int(rul.size), n_total=int(n_total),
            mean=mean, std=std, cv=(std / mean if mean else float("nan")),
            q05=float(np.quantile(rul, 0.05)),
            q25=float(np.quantile(rul, 0.25)),
            median=float(np.median(rul)),
            q75=float(np.quantile(rul, 0.75)),
            q95=float(np.quantile(rul, 0.95)),
            min=float(np.min(rul)), max=float(np.max(rul)),
        )

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Observation truncation
# --------------------------------------------------------------------------- #
def _esticol_subset(esticol: np.ndarray, k: int, selection) -> np.ndarray:
    """Pick the ESTICOL points to keep.

    ``selection`` is ``"first"`` (default: the first ``k`` rows as stored, i.e. the
    historical behaviour of the sweep), ``"last"`` (the last ``k`` points in
    chronological order) or a ``(start, stop)`` pair of indices into the
    chronologically sorted points (``k`` is ignored for the pair).
    """
    n = len(esticol)
    if selection == "first":
        return esticol[:min(k, n)]
    ordered = esticol[np.argsort(esticol[:, 0], kind="stable")]
    if selection == "last":
        return ordered[max(n - k, 0):]
    if isinstance(selection, (tuple, list)) and len(selection) == 2:
        return ordered[int(selection[0]):int(selection[1])]
    raise ValueError(f"selection must be 'first', 'last' or a (start, stop) pair, got {selection!r}")


def truncate_last_cycle_observations(dataset, k: int, selection="first"):
    """Return a shallow-copied dataset whose last cycle keeps only k ESTICOL points.

    The last cycle's ``data_with_time_division`` entry is a list of channels
    ``[esticol, etv]`` (this order is set in
    ``data_preparation.preprocess_field_data``); each is an ``(m, 2)`` array of
    ``(time, value)`` sorted by time.

    Only the **ESTICOL** channel (index 0) is truncated to its first ``k`` points;
    the **ETV** channel (index 1) is kept in full. So the sweep varies the amount
    of ESTICOL data while holding ETV fixed. The matching ``time_indices`` are
    rebuilt so the MCMC likelihood and the smoother stay consistent.

    ``k`` is clamped to the ESTICOL channel's available length. Which points are kept
    is set by ``selection`` -- see :func:`_esticol_subset` (default ``"first"``).
    """
    ds = copy.copy(dataset)                      # shallow: we replace only 2 fields
    ds.data_with_time_division = list(dataset.data_with_time_division)
    ds.time_indices = list(dataset.time_indices)

    last = len(ds.data_with_time_division) - 1
    channels = [np.asarray(c) for c in dataset.data_with_time_division[last]]

    # Truncate ESTICOL (channel 0) only; keep ETV (channel 1) full.
    trunc_channels = list(channels)
    trunc_channels[0] = _esticol_subset(channels[0], k, selection)

    # Rebuild combined sorted timeline and per-channel index sets, matching the
    # convention used in data_preparation.preprocess_field_data.
    combined = np.sort(np.concatenate(trunc_channels, axis=0), axis=0)
    new_time_indices = [
        np.where(np.isin(combined[:, 0], ch[:, 0]))[0]
        for ch in trunc_channels
    ]

    ds.data_with_time_division[last] = trunc_channels
    ds.time_indices[last] = new_time_indices
    return ds, last


# --------------------------------------------------------------------------- #
# Single-k run
# --------------------------------------------------------------------------- #
def _hours_to_index(sim_time, target_hours, name):
    """Map a time given in HOURS to the nearest index on ``sim_time`` (DAYS).

    ``sim_time`` is in days; ``target_hours`` is in hours. Rather than requiring
    an exact grid hit (which silently produced an empty ``np.where`` and a cryptic
    IndexError when the anchor was stale or in the wrong unit), snap to the
    closest grid point and warn if the gap is large.
    """
    hours = np.asarray(sim_time, dtype=float) * 24.0
    idx = int(np.argmin(np.abs(hours - target_hours)))
    gap = abs(hours[idx] - target_hours)
    if hours.size == 0:
        raise ValueError("sim_time is empty.")
    # a "large" gap = more than one grid spacing away from any point
    spacing = np.median(np.diff(hours)) if hours.size > 1 else np.inf
    if gap > max(spacing, 1.0):
        raise ValueError(
            f"{name}={target_hours} h is far from the SG time grid "
            f"(nearest grid point {hours[idx]:.0f} h, gap {gap:.0f} h). "
            f"Valid anchors (hours) run {hours.min():.0f}..{hours.max():.0f}. "
            f"This usually means the anchor is stale (from a different SG) or "
            f"the TIME.npy is in the wrong unit (must be DAYS)."
        )
    return idx


@dataclass
class PrognosticsStages:
    """Prior / BMU / BMU+EnKS trajectory ensembles and per-cycle posterior samples.

    This is everything :func:`run_observation_size` computes on its way to
    ``RULStats`` -- exposed on its own so callers can also inspect, plot, or
    save the trajectories and each cycle's posterior parameter samples, not
    just the final RUL summary.
    """

    cycle_results: list                 # one CycleResult per assimilated cycle, in order
    posterior_samples: list             # one (n_rul_samples, input_dimension) array per cycle
    bmu_full: np.ndarray                # (n_rul_samples, n_sim_time) -- full simulation-time axis
    prior_traj: np.ndarray              # (n_rul_samples, n_pc) -- on the prognostics window
    bmu_traj: np.ndarray                # (n_rul_samples, n_pc) -- on the prognostics window
    bmu_enks_traj: np.ndarray           # (n_ens, n_enks) -- on the smoother's own time axis
    pc_time: np.ndarray                 # time axis (days) for prior_traj / bmu_traj
    enks_time: np.ndarray               # time axis (days) for bmu_enks_traj

    def rul_stats(self, threshold: float, n_obs: int) -> dict[str, RULStats]:
        """RUL statistics per stage for a given crossing ``threshold``."""
        stage_data = {
            "prior": (self.prior_traj, self.pc_time, len(self.prior_traj)),
            "bmu": (self.bmu_traj, self.pc_time, len(self.bmu_traj)),
            "bmu_enks": (self.bmu_enks_traj, self.enks_time, len(self.bmu_enks_traj)),
        }
        return {
            stage: RULStats.from_sample(
                stage, n_obs, first_crossing_rul(traj, t, threshold), n_total,
            )
            for stage, (traj, t, n_total) in stage_data.items()
        }


def compute_prognostics_stages(
    pipe,
    cycle_results: list,
    dataset,
    split_sim_hours: int,
    prognostics_split_hours: int,
    n_rul_samples: int = 1000,
) -> PrognosticsStages:
    """Build the prior/BMU/BMU+EnKS trajectory ensembles from completed cycles.

    Parameters
    ----------
    pipe : CloggingPipeline
        A pipeline that has already had ``setup()`` called.
    cycle_results : list[CycleResult]
        One result per assimilated cycle, in order (as returned by
        ``pipe.run()``, or ``prior_cycle_results + [last_res]`` in the
        observation-size sweep).
    dataset : CloggingDataset
        The dataset the *last* cycle in ``cycle_results`` was assimilated
        against (``pipe.dataset`` for a full run; the ESTICOL-truncated
        dataset for an observation-size sweep) -- passed to the EnKS smoother.
    split_sim_hours, prognostics_split_hours, n_rul_samples :
        See :func:`run_observation_size`.

    Notes
    -----
    The BMU trajectory is spliced from exactly the *first two* entries of
    ``cycle_results`` at ``split_sim_hours`` (cycle 0 before the split, cycle 1
    after) -- this mirrors the original notebook's two-cycle (one cleaning)
    assumption and has not been generalized to more than two cycles. For a
    scenario with more than one cleaning (more than two cycles), only the
    first two cycles' posteriors feed the spliced BMU trajectory.
    """
    cfg = pipe.cfg
    sim_time = pipe.dataset.simulation_time

    # --- posterior samples per cycle --------------------------------------- #
    posterior_samples = [
        np.asarray([
            np.asarray(m.getSample(n_rul_samples)).reshape(-1)
            for m in res.marginals
        ]).T
        for res in cycle_results
    ]

    # --- concatenated BMU trajectory --------------------------------------- #
    nc = _hours_to_index(sim_time, split_sim_hours, "split_sim_hours")
    trajs = [pipe.base_surrogate.predict(pipe.global_scaler.transform(ps))
             for ps in posterior_samples]
    bmu_full = np.concatenate((trajs[0][:, :nc + 1], trajs[1][:, nc + 1:]), axis=1)

    # --- prognostics window ------------------------------------------------ #
    nc2 = _hours_to_index(sim_time, prognostics_split_hours, "prognostics_split_hours")
    pc_time = sim_time[nc2 + 1:]

    # prior stage
    prior_sample = np.asarray(cfg.prior.composed().getSample(n_rul_samples))
    prior_traj = pipe.base_surrogate.predict(pipe.global_scaler.transform(prior_sample))
    prior_traj = np.asarray([
        np.interp(pc_time, sim_time, prior_traj[i, :]) for i in range(len(prior_traj))
    ])

    # BMU stage (posterior, no smoothing) on the prognostics window
    bmu_traj = np.asarray([
        np.interp(pc_time, sim_time, bmu_full[i, :]) for i in range(len(bmu_full))
    ])

    # BMU+EnKS stage (posterior + ensemble Kalman smoothing)
    # smooth_cycles returns its own (k-dependent) time axis, which interleaves
    # observation timestamps with the simulation grid and is generally longer
    # than sim_time. Both sim_time and the smoother's axis are in DAYS, while
    # prognostics_split_hours is in HOURS -- convert, then slice by time value
    # (not by simulation-grid index).
    cti = pipe.cleaning_time_indices()
    smoothed, smoothed_time = smooth_cycles(bmu_full, dataset, cfg, cti)
    prog_days = prognostics_split_hours / 24.0
    enks_mask = smoothed_time > prog_days
    bmu_enks_traj = smoothed[:, enks_mask]
    enks_time = smoothed_time[enks_mask]

    return PrognosticsStages(
        cycle_results=cycle_results,
        posterior_samples=posterior_samples,
        bmu_full=bmu_full,
        prior_traj=prior_traj,
        bmu_traj=bmu_traj,
        bmu_enks_traj=bmu_enks_traj,
        pc_time=pc_time,
        enks_time=enks_time,
    )


def _threshold(cfg) -> float:
    if cfg.clogging_threshold is None:
        raise ValueError("cfg.clogging_threshold is not set: use clogging_prognostics.load_thresholds().")
    return cfg.clogging_threshold


def _assimilate_observation_size(
    pipe,
    k: int,
    split_sim_hours: int,
    prognostics_split_hours: int,
    n_rul_samples: int,
    selection="first",
) -> tuple[list, "PrognosticsStages"]:
    """Shared body of :func:`run_observation_size` / :func:`run_observation_size_with_stages`.

    Returns ``(cycle_results, stages)`` -- see :func:`run_full_methodology` for
    the full-data analogue. Kept private: callers should use one of the two
    public wrappers below.
    """
    cfg = pipe.cfg
    dataset, last = truncate_last_cycle_observations(pipe.dataset, k, selection)

    ot.RandomGenerator.SetSeed(cfg.assimilation.random_seed)

    # --- earlier cycles are k-independent: compute once, then reuse -------- #
    # Only the last cycle's ESTICOL data varies with k, so cycles 0..last-1
    # are assimilated a single time (cached on the pipeline) and shared across
    # the whole observation-size sweep.
    if getattr(pipe, "prior_cycle_results_", None) is None:
        pipe.precompute_prior_cycles()
    prior_cycle_results = pipe.prior_cycle_results_

    # --- assimilate ONLY the last cycle, with the truncated observations --- #
    # Reseed here so every k starts the last-cycle assimilation from an
    # identical RNG state: the only thing that differs across the sweep is the
    # amount of ESTICOL data, not the Monte Carlo draws.
    ot.RandomGenerator.SetSeed(cfg.assimilation.random_seed)
    X0, y0, _ = build_doe(pipe.dataset, cfg)
    last_res = assimilate_cycle(
        cycle_index=last,
        dataset=dataset,                     # ESTICOL-truncated (last cycle only)
        cfg=cfg,
        base_surrogate=pipe.base_surrogate,
        global_scaler=pipe.global_scaler,
        per_variable_scalers=pipe.per_variable_scalers,
        X0=X0, y0=y0,
    )
    cycle_results = prior_cycle_results + [last_res]

    stages = compute_prognostics_stages(
        pipe, cycle_results, dataset,
        split_sim_hours, prognostics_split_hours, n_rul_samples,
    )
    return cycle_results, stages


def run_observation_size(
    pipe,
    k: int,
    split_sim_hours: int = 176568,
    prognostics_split_hours: int = 176808,
    n_rul_samples: int = 1000,
    selection="first",
) -> list[RULStats]:
    """Run the last-cycle pipeline with ``k`` observations; return RUL stats.

    Parameters
    ----------
    pipe : CloggingPipeline
        A pipeline that has already had ``setup()`` called (base surrogate,
        scalers, and full field data available).
    k : int
        Number of field-data observations to retain in the last cycle.
    split_sim_hours : int
        Simulation-time (hours) at which the CC/PC trajectories are concatenated
        (the notebook's ``176568`` -> index ``nc``).
    prognostics_split_hours : int
        Simulation-time (hours) marking the start of the prognostics window
        (the notebook's ``176808`` -> index ``nc2``).
    n_rul_samples : int
        Number of posterior/prior MC samples used to build each RUL ensemble.

    Returns
    -------
    list[RULStats]
        One entry per stage: prior, BMU, BMU+EnKS.
    """
    _, stages = _assimilate_observation_size(
        pipe, k, split_sim_hours, prognostics_split_hours, n_rul_samples, selection,
    )
    return list(stages.rul_stats(_threshold(pipe.cfg), k).values())


def run_observation_size_with_stages(
    pipe,
    k: int,
    split_sim_hours: int = 176568,
    prognostics_split_hours: int = 176808,
    n_rul_samples: int = 1000,
    selection="first",
) -> tuple[list[RULStats], list, "PrognosticsStages"]:
    """Like :func:`run_observation_size`, but also returns the per-cycle
    diagnostics/posteriors and the prior/BMU/BMU+EnKS trajectory ensembles.

    Returns ``(stats, cycle_results, stages)`` -- identical ``stats`` to
    :func:`run_observation_size` (same RNG path, same numbers), plus
    ``cycle_results`` (one ``CycleResult`` per cycle: active variable,
    convergence, posterior marginals) and ``stages`` (the ``PrognosticsStages``
    trajectory ensembles), for callers that want to save or plot them rather
    than only the summary statistics.
    """
    cycle_results, stages = _assimilate_observation_size(
        pipe, k, split_sim_hours, prognostics_split_hours, n_rul_samples, selection,
    )
    stats = list(stages.rul_stats(_threshold(pipe.cfg), k).values())
    return stats, cycle_results, stages


def run_full_methodology(
    pipe,
    split_sim_hours: int = 176568,
    prognostics_split_hours: int = 176808,
    n_rul_samples: int = 1000,
) -> tuple[list, PrognosticsStages]:
    """Assimilate every cycle on its full (untruncated) field data.

    The single-run counterpart of the ``k``-sweep in :func:`run_observation_size`,
    for when there is no observation-size sweep to run -- e.g. one full,
    all-data methodology run per steam generator. Returns
    ``(cycle_results, stages)`` so every cycle's diagnostics (active variable,
    convergence, posterior samples) and the resulting prior/BMU/BMU+EnKS
    trajectory ensembles are all available to the caller.
    """
    cfg = pipe.cfg
    ot.RandomGenerator.SetSeed(cfg.assimilation.random_seed)
    cycle_results = pipe.run()
    stages = compute_prognostics_stages(
        pipe, cycle_results, pipe.dataset,
        split_sim_hours, prognostics_split_hours, n_rul_samples,
    )
    return cycle_results, stages