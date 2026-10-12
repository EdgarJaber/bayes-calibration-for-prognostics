#!/usr/bin/env python3
"""ESTICOL "kinetics" experiment on SG B.

Instead of the first-k sweep of ``esticol_sweep.py``, assimilate only a
sub-population of the LAST cycle's ESTICOL points (TVE data held fixed), chosen
by the clogging kinetics they follow:

  quasi_linear : the first 20 ESTICOL points after the last chemical cleaning
  exponential  : the last 20 ESTICOL points, counting backwards from the last one

Everything else (DoE, HSIC screening, 15-member calibration ensemble, fixed-Sobol
QMC weights, MCMC, EnKS, RUL stages) is the standard pipeline. The earlier,
selection-independent cycles are assimilated once and shared by both experiments.

Config: configs/esticol_kinetics_config.json (each experiment has a name, a ``selection``
-- "first", "last" or a [start, stop] pair of chronological indices into the last
cycle's ESTICOL points -- and ``k``, the number of points for first/last).

    python scripts/esticol_kinetics.py --config configs/esticol_kinetics_config.json --outdir results_kinetics
    python scripts/esticol_kinetics.py --dry-run            # only list the selected points (fast)
    python scripts/esticol_kinetics.py --only exponential   # run a single experiment

Outputs, in --outdir:
    esticol_kinetics.csv / .json / .tex           RUL statistics per experiment and stage
    esticol_points/<SG>_<name>_esticol_points.csv  the ESTICOL points actually assimilated
    posteriors/<SG>_<name>_cycle<i>_posterior.csv  posterior parameter samples per window
    trajectories/<SG>_<name>_<stage>.npy (+ _time.npy)  prior/bmu/bmu_enks ensembles
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.abspath(os.path.join(HERE, '..'))                       # applications/sg_clogging
sys.path.insert(0, APP)                                               # clogging_prognostics
sys.path.insert(0, os.path.abspath(os.path.join(APP, '..', '..')))    # bayes_fusion (repo root)

from clogging_prognostics import (CloggingPipeline, PipelineConfig, load_thresholds,
                                  run_observation_size_with_stages, threshold_symbol)
from clogging_prognostics.data_io import load_dataset
from clogging_prognostics.experiment import truncate_last_cycle_observations

STAGE_ORDER = ["prior", "bmu", "bmu_enks"]


def parse_selection(exp):
    sel = exp["selection"]
    return tuple(sel) if isinstance(sel, list) else sel


def selected_points(dataset, exp):
    """The ESTICOL points of the last cycle that this experiment assimilates."""
    truncated, last = truncate_last_cycle_observations(dataset, exp.get("k", 0), parse_selection(exp))
    return np.asarray(truncated.data_with_time_division[last][0])


def save_points(outdir, tag, points):
    d = os.path.join(outdir, "esticol_points")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{tag}_esticol_points.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_days", "tau_c"])
        w.writerows(np.asarray(points).tolist())
    return path


def save_posteriors(outdir, tag, cfg, cycle_results, n_samples):
    d = os.path.join(outdir, "posteriors")
    os.makedirs(d, exist_ok=True)
    names = list(cfg.prior.names)
    paths = []
    for res in cycle_results:
        samples = np.asarray([np.asarray(m.getSample(n_samples)).reshape(-1)
                              for m in res.marginals]).T
        path = os.path.join(d, f"{tag}_cycle{res.cycle_index}_posterior.csv")
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(names)
            w.writerows(samples.tolist())
        paths.append(path)
    return paths


def save_trajectories(outdir, tag, stages):
    d = os.path.join(outdir, "trajectories")
    os.makedirs(d, exist_ok=True)
    arrays = {"prior": (stages.prior_traj, stages.pc_time),
              "bmu": (stages.bmu_traj, stages.pc_time),
              "bmu_enks": (stages.bmu_enks_traj, stages.enks_time)}
    paths = []
    for stage, (traj, t) in arrays.items():
        tp, tt = (os.path.join(d, f"{tag}_{stage}.npy"), os.path.join(d, f"{tag}_{stage}_time.npy"))
        np.save(tp, traj)
        np.save(tt, np.asarray(t))
        paths += [tp, tt]
    return paths


def build_pipeline(common, sg):
    cfg = PipelineConfig(
        base_directory=common["base_directory"],
        input_dimension=common.get("input_dimension", 7),
        n_doe_samples=common.get("n_doe_samples", 1000),
        n_cycles=sg["n_cycles"],
        clogging_threshold=load_thresholds(common["base_directory"])[sg["sg_tag"]],
    )
    cfg.mcmc.n_chains = common.get("n_chains", 3)
    cfg.mcmc.nb_weights = common.get("nb_weights", 1024)
    cfg.mcmc.sample_size = common.get("sample_size", 1000)
    cfg.mcmc.burn_in = common.get("burn_in", 100)
    return CloggingPipeline(cfg).setup(sg_tag=sg["sg_tag"])


def latex_table(rows, sg_display, label, symbol, unit):
    def f(x):
        return "--" if (x != x) else f"{x:.1f}"
    L = [r"\begin{table}[ht]", r"  \centering",
         (rf"  \caption{{RUL mean and standard deviation (in {unit}) when only a sub-population of the "
          rf"ESTICOL points of the last cycle is assimilated, for \textbf{{{sg_display}}} "
          rf"(threshold ${symbol}$, TVE data fixed).}}"),
         rf"  \label{{{label}}}", r"  \begin{tabular}{lrrrrr}", r"    \toprule",
         r"    & & \multicolumn{2}{c}{BMU} & \multicolumn{2}{c}{BMU\,+\,EnKS}\\",
         r"    \cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"    Experiment & $n$ & Mean & Std & Mean & Std \\", r"    \midrule"]
    for name, n_pts, by_stage in rows:
        b, e = by_stage["bmu"], by_stage["bmu_enks"]
        L.append(f"    {name.replace('_', ' ')} & {n_pts} & {f(b.mean)} & {f(b.std)} & {f(e.mean)} & {f(e.std)} \\\\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(APP, "configs", "esticol_kinetics_config.json"))
    ap.add_argument("--outdir", default="results_kinetics")
    ap.add_argument("--only", nargs="*", help="run only these experiment names")
    ap.add_argument("--dry-run", action="store_true", help="only list/save the selected ESTICOL points")
    args = ap.parse_args()

    with open(args.config) as fh:
        conf = json.load(fh)
    common, sg, experiments = conf["common"], conf["steam_generator"], conf["experiments"]
    if args.only:
        experiments = [e for e in experiments if e["name"] in args.only]
    os.makedirs(args.outdir, exist_ok=True)
    sg_tag = sg["sg_tag"]                     # DATA/ file prefix and output filename tag
    alias = sg_tag
    print(f"=== ESTICOL kinetics experiment: {sg['display']}, "
          f"nb_weights={common.get('nb_weights', 1024)} ===", flush=True)

    if args.dry_run:
        dataset = load_dataset(common["base_directory"], sg_tag)
        for exp in experiments:
            pts = selected_points(dataset, exp)
            path = save_points(args.outdir, f"{alias}_{exp['name']}", pts)
            print(f"  {exp['name']}: {len(pts)} points, t in [{pts[:, 0].min():.0f}, {pts[:, 0].max():.0f}] d, "
                  f"tau_c in [{pts[:, 1].min():.1f}, {pts[:, 1].max():.1f}] -> {path}")
        return 0

    pipe = build_pipeline(common, sg)
    n_rul_samples = common.get("n_rul_samples", 1000)
    unit = common.get("unit", "days")

    csv_rows, json_out, table_rows = [], {}, []
    for exp in experiments:
        name, selection = exp["name"], parse_selection(exp)
        tag = f"{alias}_{name}"
        t0 = time.time()
        pts = selected_points(pipe.dataset, exp)
        print(f"\n--- {name}: {exp.get('description', '')} ({len(pts)} points, "
              f"t in [{pts[:, 0].min():.0f}, {pts[:, 0].max():.0f}] d) ---", flush=True)
        save_points(args.outdir, tag, pts)
        try:
            stats, cycle_results, stages = run_observation_size_with_stages(
                pipe, k=exp.get("k", len(pts)),
                split_sim_hours=sg["split_sim_hours"],
                prognostics_split_hours=sg["prognostics_split_hours"],
                n_rul_samples=n_rul_samples, selection=selection,
            )
        except Exception as e:
            print(f"  {name}: FAILED: {e}", flush=True)
            traceback.print_exc()
            continue
        for p in (save_posteriors(args.outdir, tag, pipe.cfg, cycle_results, n_rul_samples)
                  + save_trajectories(args.outdir, tag, stages)):
            print(f"    wrote {p}", flush=True)
        by_stage = {s.stage: s for s in stats}
        table_rows.append((name, len(pts), by_stage))
        json_out[name] = {"n_points": len(pts), "selection": exp["selection"], "k": exp.get("k"),
                          **{st: by_stage[st].to_dict() for st in STAGE_ORDER}}
        for st in STAGE_ORDER:
            s = by_stage[st]
            csv_rows.append(dict(experiment=name, n_points=len(pts), stage=st, mean=s.mean, std=s.std, cv=s.cv,
                                 q05=s.q05, median=s.median, q95=s.q95, count=s.count, n_total=s.n_total))
        print(f"  {name}: BMU mean={by_stage['bmu'].mean:8.1f} std={by_stage['bmu'].std:7.1f} | "
              f"EnKS mean={by_stage['bmu_enks'].mean:8.1f} std={by_stage['bmu_enks'].std:7.1f} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    cols = ["experiment", "n_points", "stage", "mean", "std", "cv", "q05", "median", "q95", "count", "n_total"]
    with open(os.path.join(args.outdir, "esticol_kinetics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(csv_rows)
    with open(os.path.join(args.outdir, "esticol_kinetics.json"), "w") as fh:
        json.dump(json_out, fh, indent=2)
    if table_rows:
        with open(os.path.join(args.outdir, "esticol_kinetics.tex"), "w") as fh:
            fh.write(latex_table(table_rows, sg["display"], sg.get("label", "tab:esticol_kinetics"),
                                 threshold_symbol(sg["sg_tag"]), unit) + "\n")
    print(f"\nwrote {args.outdir}/esticol_kinetics.{{csv,json,tex}}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
