"""Steam-generator clogging prognostics: application of :mod:`bayes_fusion`.

>>> from clogging_prognostics import PipelineConfig, CloggingPipeline
>>> cfg = PipelineConfig(base_directory="DATA/", n_cycles=2)
>>> pipe = CloggingPipeline(cfg).setup(sg_tag="SGB")
>>> results = pipe.run()
"""

from bayes_fusion.prognostics import (bootstrap_align, compute_prognostics_trajectories,
                                      first_crossing_rul)
from bayes_fusion.screening import estimate_kl, kl_divergence, select_active_index
from bayes_fusion.smoothing import ensemble_kalman_smoother
from bayes_fusion.surrogates import build_base_surrogate, build_calibration_ensemble

from .assimilation import CycleResult, assimilate_cycle
from .config import (AssimilationConfig, MCMCConfig, MetamodelConfig, PipelineConfig,
                     PriorSpec, load_thresholds, threshold_symbol)
from .data_io import CloggingDataset, load_dataset
from .experiment import (PrognosticsStages, RULStats, compute_prognostics_stages,
                         run_full_methodology, run_observation_size,
                         run_observation_size_with_stages,
                         truncate_last_cycle_observations)
from .pipeline import CloggingPipeline
from .smoothing import smooth_cycles

__all__ = [
    "PipelineConfig", "PriorSpec", "MCMCConfig", "AssimilationConfig", "MetamodelConfig",
    "load_thresholds", "threshold_symbol",
    "CloggingDataset", "load_dataset", "CloggingPipeline",
    "CycleResult", "assimilate_cycle",
    "select_active_index", "kl_divergence", "estimate_kl",
    "build_base_surrogate", "build_calibration_ensemble",
    "ensemble_kalman_smoother", "smooth_cycles",
    "first_crossing_rul", "compute_prognostics_trajectories", "bootstrap_align",
    "RULStats", "PrognosticsStages", "compute_prognostics_stages",
    "run_full_methodology", "run_observation_size", "run_observation_size_with_stages",
    "truncate_last_cycle_observations",
]
