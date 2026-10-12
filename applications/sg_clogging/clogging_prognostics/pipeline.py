"""Top-level orchestration of the clogging prognostics methodology.

The notebook's scenario loop is replaced by :meth:`CloggingPipeline.run`, which
loops over cycles and calls :func:`assimilation.assimilate_cycle` for each. All
state (dataset, base surrogate, scalers, per-cycle results) is held on the
pipeline object so downstream analysis stays organized and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import openturns as ot
from sklearn.model_selection import train_test_split

from .assimilation import CycleResult, assimilate_cycle
from .data_io import (CloggingDataset, build_doe,
                      build_per_variable_scalers, load_dataset)
from bayes_fusion.surrogates import build_base_surrogate


@dataclass
class CloggingPipeline:
    """End-to-end pipeline for hybrid clogging prognostics."""

    cfg: object                                   # PipelineConfig

    dataset: CloggingDataset = field(init=False, default=None)
    base_surrogate: object = field(init=False, default=None)
    global_scaler: object = field(init=False, default=None)
    per_variable_scalers: list = field(init=False, default=None)
    X: np.ndarray = field(init=False, default=None)
    y: np.ndarray = field(init=False, default=None)
    cycle_results: list[CycleResult] = field(init=False, default_factory=list)
    # cache of the k-independent earlier-cycle assimilation results (cycles
    # 0 .. n_cycles-2), populated by precompute_prior_cycles(); reused across
    # every k in the observation-size sweep so only the last cycle re-runs.
    prior_cycle_results_: list = field(init=False, default=None)

    # -- setup -------------------------------------------------------------- #
    def setup(self, sg_tag: str = "SGB") -> "CloggingPipeline":
        """Load data, build the DoE, scalers, and the base surrogate."""
        self.dataset = load_dataset(self.cfg.base_directory, sg_tag=sg_tag)

        self.X, self.y, self.global_scaler = build_doe(self.dataset, self.cfg)
        self.per_variable_scalers = build_per_variable_scalers(self.dataset, self.cfg)

        X_train, _, y_train, _ = train_test_split(
            self.X, self.y, train_size=self.cfg.train_size, random_state=42,
        )
        self.base_surrogate = build_base_surrogate(
            X_train, y_train, self.cfg, self.dataset.simulation_time,
        )
        return self

    # -- observation-sweep cache ------------------------------------------- #
    def precompute_prior_cycles(self) -> list[CycleResult]:
        """Assimilate the k-independent earlier cycles once and cache them.

        In the observation-size sweep only the *last* cycle's ESTICOL data is
        truncated by k; cycles ``0 .. n_cycles-2`` see their full observations
        for every k, so their assimilation is identical across the sweep. This
        runs them a single time (on the full, untruncated dataset) and stores
        the results in ``prior_cycle_results_`` for reuse.

        Returns the cached list (empty when ``n_cycles == 1``).
        """
        if self.base_surrogate is None:
            raise RuntimeError("Call setup() before precompute_prior_cycles().")

        ot.RandomGenerator.SetSeed(self.cfg.assimilation.random_seed)
        results = []
        for cycle_index in range(self.cfg.n_cycles - 1):     # all but the last
            X0, y0, _ = build_doe(self.dataset, self.cfg)
            res = assimilate_cycle(
                cycle_index=cycle_index,
                dataset=self.dataset,                        # full, untruncated
                cfg=self.cfg,
                base_surrogate=self.base_surrogate,
                global_scaler=self.global_scaler,
                per_variable_scalers=self.per_variable_scalers,
                X0=X0, y0=y0,
            )
            results.append(res)
        self.prior_cycle_results_ = results
        return results

    # -- scenario loop ------------------------------------------------------ #
    def run(self) -> list[CycleResult]:
        """Assimilate every clogging cycle. Returns the per-cycle results."""
        if self.base_surrogate is None:
            raise RuntimeError("Call setup() before run().")

        ot.RandomGenerator.SetSeed(self.cfg.assimilation.random_seed)
        self.cycle_results = []

        for cycle_index in range(self.cfg.n_cycles):
            print(f"Starting assimilation for cycle {cycle_index + 1}")
            # fresh DoE per cycle (matches the notebook's per-iteration reset)
            X0, y0, _ = build_doe(self.dataset, self.cfg)
            result = assimilate_cycle(
                cycle_index=cycle_index,
                dataset=self.dataset,
                cfg=self.cfg,
                base_surrogate=self.base_surrogate,
                global_scaler=self.global_scaler,
                per_variable_scalers=self.per_variable_scalers,
                X0=X0,
                y0=y0,
            )
            print(
                f"  cycle {cycle_index + 1}: converged={result.converged} "
                f"in {result.n_iterations} iters, active var index={result.active_index}"
            )
            self.cycle_results.append(result)

        return self.cycle_results

    # -- posterior sampling helpers ---------------------------------------- #
    def posterior_dataframes(self, n_samples: int = 1000):
        """Sample each cycle's refined marginals into a list of DataFrames."""
        import pandas as pd

        cols = [f"X{j}" for j in range(1, self.cfg.input_dimension + 1)]
        dfs = []
        for res in self.cycle_results:
            samples = np.asarray([
                np.asarray(m.getSample(n_samples)).reshape(-1)
                for m in res.marginals
            ])
            dfs.append(pd.DataFrame(samples.T, columns=cols))
        return dfs

    def cleaning_time_indices(self) -> np.ndarray:
        """Indices of the cleaning dates within the simulation time grid."""
        sim_time = self.dataset.simulation_time
        dates = np.asarray(self.dataset.cleaning_dates).astype(int)
        return np.asarray([
            np.where(sim_time == dates[i])[0][0] for i in range(len(dates))
        ])