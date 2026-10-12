"""Clogging-specific entry point to the generic calibration loop.

Fixes the only application-level choice of :func:`bayes_fusion.assimilation.assimilate_cycle`:
the simulation grid is in days and the calibration surrogates are built on an hourly axis.
"""

from __future__ import annotations

from bayes_fusion.assimilation import CycleResult
from bayes_fusion.assimilation import assimilate_cycle as _assimilate_cycle

DAYS_TO_HOURS = 24.0

__all__ = ["CycleResult", "assimilate_cycle", "DAYS_TO_HOURS"]


def assimilate_cycle(*args, **kwargs) -> CycleResult:
    kwargs.setdefault("time_scale", DAYS_TO_HOURS)
    return _assimilate_cycle(*args, **kwargs)
