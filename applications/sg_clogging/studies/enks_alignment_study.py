#!/usr/bin/env python3
"""Impact of the EnKS observation-alignment issue on the BMU+EnKS RUL statistics.

``smooth_cycles(..., align_observations=False)`` (the original behaviour, used for the published
numbers) updates the i-th column of each cycle's time axis with the i-th observation of a
channel, instead of the column holding the observation's own time. ``align_observations=True``
assimilates every observation at its own time.

The study rebuilds, from the saved posterior samples (``results/`` and ``results_sweep/``), the
spliced BMU trajectories of every run, smooths them both ways with the *same* ensemble, and
compares the RUL statistics on the prognostics window. The "published" rows are also checked
against the BMU+EnKS trajectories saved by the experiments (they differ only by the Monte-Carlo
draw of the parameters).

    python studies/enks_alignment_study.py --results results --sweep results_sweep --out results_enks_study
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))                 # clogging_prognostics
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))     # bayes_fusion

from bayes_fusion.prognostics import first_crossing_rul
from clogging_prognostics import CloggingPipeline, PipelineConfig, smooth_cycles
from clogging_prognostics.experiment import (RULStats, _hours_to_index,
                                             truncate_last_cycle_observations)


def build_pipeline(common, sg):
    cfg = PipelineConfig(base_directory=common["base_directory"],
                         input_dimension=common.get("input_dimension", 7),
                         n_doe_samples=common.get("n_doe_samples", 1000),
                         n_cycles=sg["n_cycles"], clogging_threshold=sg["threshold"])
    return CloggingPipeline(cfg).setup(sg_tag=sg["sg_tag"])


def read_posteriors(directory, prefix, names, n_cycles):
    out = []
    for c in range(2):                       # the BMU splice only uses the first two cycles
        df = pd.read_csv(os.path.join(directory, "posteriors", f"{prefix}_cycle{c}_posterior.csv"))
        out.append(df[list(names)].to_numpy())
    return out


def stats_row(label, rul, n_total, rmse=(np.nan, np.nan)):
    s = RULStats.from_sample(label, 0, rul, n_total)
    return dict(stage=label, mean=s.mean, std=s.std, q05=s.q05, median=s.median, q95=s.q95,
                crossing=s.count / n_total, rmse_esticol=rmse[0], rmse_tve=rmse[1])


def misfit(time, mean_traj, dataset, t_start):
    """RMSE (in % of clogging) between a mean trajectory and the last cycle's observations."""
    out = []
    for ch in dataset.data_with_time_division[-1]:
        ch = np.asarray(ch)
        ch = ch[ch[:, 0] >= t_start]
        out.append(float(np.sqrt(np.mean((np.interp(ch[:, 0], time, mean_traj) - ch[:, 1]) ** 2)))
                   if len(ch) else np.nan)
    return tuple(out)


def analyse(pipe, dataset, posteriors, split_hours, prog_hours, threshold):
    sim_time = pipe.dataset.simulation_time
    nc = _hours_to_index(sim_time, split_hours, "split_sim_hours")
    nc2 = _hours_to_index(sim_time, prog_hours, "prognostics_split_hours")
    trajs = [pipe.base_surrogate.predict(pipe.global_scaler.transform(ps)) for ps in posteriors]
    bmu_full = np.concatenate((trajs[0][:, :nc + 1], trajs[1][:, nc + 1:]), axis=1)
    pc_time = sim_time[nc2 + 1:]
    n = len(bmu_full)

    rows = []
    bmu = np.asarray([np.interp(pc_time, sim_time, bmu_full[i]) for i in range(n)])
    t_start = prog_hours / 24.0
    rows.append(stats_row("BMU", first_crossing_rul(bmu, pc_time, threshold), n,
                          misfit(sim_time, bmu_full.mean(axis=0), dataset, t_start)))
    cti = pipe.cleaning_time_indices()
    for label, aligned in (("BMU+EnKS (published)", False), ("BMU+EnKS (aligned)", True)):
        sm, t = smooth_cycles(bmu_full, dataset, pipe.cfg, cti, align_observations=aligned)
        mask = t > t_start
        order = np.argsort(t[mask], kind="stable")
        rows.append(stats_row(label, first_crossing_rul(sm[:, mask], t[mask], threshold), n,
                              misfit(t[mask][order], sm[:, mask].mean(axis=0)[order], dataset, t_start)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rul-config", default=os.path.join(HERE, "..", "configs", "rul_tables_config.json"))
    ap.add_argument("--sweep-config", default=os.path.join(HERE, "..", "configs", "esticol_sweep_config.json"))
    ap.add_argument("--results", default="results")
    ap.add_argument("--sweep", default="results_sweep")
    ap.add_argument("--out", default="results_enks_study")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rul_conf = json.load(open(args.rul_config))
    sweep_conf = json.load(open(args.sweep_config))
    sweep_by_tag = {s["sg_tag"]: s for s in sweep_conf["steam_generators"]}

    all_rows = []
    for sg in rul_conf["steam_generators"]:
        tag = sg["sg_tag"]
        print(f"=== {sg['display']}: fitting the base surrogate ===", flush=True)
        pipe = build_pipeline(rul_conf["common"], sg)
        names = pipe.cfg.prior.names

        post = read_posteriors(args.results, tag, names, sg["n_cycles"])
        rows = analyse(pipe, pipe.dataset, post, sg["split_sim_hours"], sg["prognostics_split_hours"],
                       sg["threshold"])
        saved = os.path.join(args.results, "trajectories")
        t_saved = np.load(os.path.join(saved, f"{tag}_bmu_enks_time.npy"))
        traj_saved = np.load(os.path.join(saved, f"{tag}_bmu_enks.npy"))
        r_saved = first_crossing_rul(traj_saved, t_saved, sg["threshold"])
        rows.append(stats_row("BMU+EnKS (saved by the experiment)", r_saved, len(traj_saved)))
        all_rows += [dict(sg=sg["display"], run="full data", k=None, **r) for r in rows]

        ssg = sweep_by_tag.get(tag)
        if ssg is None or not os.path.isdir(args.sweep):
            continue
        for k in ssg["obs_sizes"]:
            ds, _ = truncate_last_cycle_observations(pipe.dataset, k)
            post = read_posteriors(args.sweep, f"{tag}_k{k:04d}", names, ssg["n_cycles"])
            rows = analyse(pipe, ds, post, ssg["split_sim_hours"], ssg["prognostics_split_hours"],
                           ssg["threshold"])
            t_saved = np.load(os.path.join(args.sweep, "trajectories", f"{tag}_k{k:04d}_bmu_enks_time.npy"))
            traj_saved = np.load(os.path.join(args.sweep, "trajectories", f"{tag}_k{k:04d}_bmu_enks.npy"))
            r_saved = first_crossing_rul(traj_saved, t_saved, ssg["threshold"])
            rows.append(stats_row("BMU+EnKS (saved by the experiment)", r_saved, len(traj_saved)))
            all_rows += [dict(sg=ssg["display"], run="sweep", k=k, **r) for r in rows]
        print(f"{sg['display']} done", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(args.out, "enks_alignment_study.csv"), index=False)
    pd.set_option("display.width", 200)
    print(df.round(1).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
