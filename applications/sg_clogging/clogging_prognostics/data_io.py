"""Loading of raw THYC / DEPOTHYC data and derived preprocessing.

Wraps the file-system access (`.npy`, `.json`, `.csv`) that used to sit in the
first notebook cell, plus the per-variable scaler construction, behind a small
:class:`CloggingDataset` container.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler

from .data_preparation import prepro_data_kl, preprocess_field_data


@dataclass
class CloggingDataset:
    """All raw and lightly-processed inputs the pipeline needs."""

    simulation_time: np.ndarray
    input_dic: dict
    cleaning_dates: list[float]
    csv_file: str

    mc_hl: np.ndarray
    mc_cl: np.ndarray
    n_max: int

    # field data, in both forms produced by preprocess_field_data
    data_without_time_division: np.ndarray
    data_with_time_division: list
    time_indices: list


def load_dataset(base_directory: str, sg_tag: str = "SGB") -> CloggingDataset:
    """Load every raw artifact for one steam generator.

    Parameters
    ----------
    base_directory : str
        Directory containing the ``{sg_tag}_*.npy/.json/.csv`` files.
    sg_tag : str
        Steam-generator identifier prefix (default ``"SGB"``).
    """
    p = lambda name: os.path.join(base_directory, f"{sg_tag}_{name}")

    simulation_time = np.load(p("TIME.npy"), allow_pickle=True)
    input_dic = json.load(open(p("METADATA.json")))
    csv_file = p("FIELD_DATA.csv")

    cleaning = input_dic["chemistry"]["cleaning"]
    cleaning_dates = [cleaning[i]["time"] / 24 for i in range(len(cleaning))]

    mc_hl = np.load(p("MC_UNIFORM_HL.npy"), allow_pickle=True)
    mc_cl = np.load(p("MC_UNIFORM_CL.npy"), allow_pickle=True)
    n_max = mc_hl.shape[0]

    data_without_time_division, _ = preprocess_field_data(
        csv_file=csv_file,
        operation_time=simulation_time,
        cleaning_dates=cleaning_dates,
        with_time_division=False,
    )
    data_with_time_division, time_indices = preprocess_field_data(
        csv_file=csv_file,
        operation_time=simulation_time,
        cleaning_dates=cleaning_dates,
        with_time_division=True,
    )

    return CloggingDataset(
        simulation_time=simulation_time,
        input_dic=input_dic,
        cleaning_dates=cleaning_dates,
        csv_file=csv_file,
        mc_hl=mc_hl,
        mc_cl=mc_cl,
        n_max=n_max,
        data_without_time_division=data_without_time_division,
        data_with_time_division=data_with_time_division,
        time_indices=time_indices,
    )


def build_doe(dataset: CloggingDataset, cfg) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Return the scaled design of experiments ``(X, y, global_scaler)``."""
    return prepro_data_kl(
        dataset.mc_hl, dataset.mc_cl,
        dataset.n_max, cfg.n_doe_samples, cfg.input_dimension,
        scale=True,
    )


def build_per_variable_scalers(dataset: CloggingDataset, cfg) -> list[StandardScaler]:
    """Fit one :class:`StandardScaler` per input variable on the *unscaled* DoE.

    These are the marginal scalers passed to the calibration metamodels (one per
    active variable), as opposed to the single global scaler used by the base
    surrogate.
    """
    x, _ = prepro_data_kl(
        dataset.mc_hl, dataset.mc_cl,
        dataset.n_max, cfg.n_doe_samples, cfg.input_dimension,
        scale=False,
    )
    scalers = []
    for i in range(x.shape[1]):
        scaler = StandardScaler()
        scaler.fit(x[:, i].reshape(-1, 1))
        scalers.append(scaler)
    return scalers
