#!/usr/bin/env python3
"""ESTICOL observation-size sweep, for every steam generator in one run.

Varies the number k of ESTICOL points assimilated on the LAST cycle (TVE data
held fixed) and records the RUL distribution statistics at each stage
(Prior / BMU / BMU+EnKS) for every k. This is the quantitative "value of
ESTICOL" study.

Config is read from JSON (see esticol_sweep_config.json): one "common" block of
shared pipeline/MCMC settings, plus a "steam_generators" list -- each entry has
its own sg_tag, window anchors, and obs_sizes grid (the two SGs' last cycles
have very different ESTICOL counts, so each grid is sized to its own SG's true
max). One invocation loops over every SG in the list, sequentially. Usage:

    python run_esticol_sweep.py --config esticol_sweep_config.json --outdir results_sweep

Outputs, in --outdir:
    esticol_sweep.csv     tidy long table: one row per (sg_tag, k, stage)
    esticol_sweep.tex     one LaTeX table per SG: RUL mean/std vs k, BMU and BMU+EnKS
    esticol_sweep.json    raw statistics per sg_tag / k / stage
    posteriors/<SG>_k<k>_cycle<i>_posterior.csv   posterior parameter samples,
                          one file per (SG, k, window) -- self-contained per k,
                          so the earlier (k-independent) cycles' posteriors do
                          get re-written for every k rather than shared once
    trajectories/<SG>_k<k>_<stage>.npy            prior/bmu/bmu_enks trajectory
                          ensembles for that (SG, k)
    trajectories/<SG>_k<k>_<stage>_time.npy       matching time axis (days)

Filenames are already sg_tag-prefixed, so every SG's output lives side by side
in the same --outdir without collisions. Within one SG, the earlier
(k-independent) assimilation windows are computed once and reused across all
of that SG's k values -- only the last cycle is re-run per k.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                    # clogging_prognostics
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))   # bayes_fusion (repo root)

from clogging_prognostics import (CloggingPipeline, PipelineConfig,
                                  run_observation_size_with_stages)

STAGE_ORDER = ["prior", "bmu", "bmu_enks"]
STAGE_LABEL = {"prior": "Prior", "bmu": "BMU", "bmu_enks": r"BMU\,+\,EnKS"}


def save_posteriors(outdir, sg_tag, k, cfg, cycle_results, n_samples):
    """Write each cycle's posterior parameter samples for this k to its own CSV.

    Plain csv module (no pandas dependency), matching the rest of this script.
    """
    posteriors_dir = os.path.join(outdir, "posteriors")
    os.makedirs(posteriors_dir, exist_ok=True)
    names = list(cfg.prior.names)
    paths = []
    for res in cycle_results:
        samples = np.asarray([
            np.asarray(m.getSample(n_samples)).reshape(-1) for m in res.marginals
        ]).T                                    # shape (n_samples, input_dimension)
        path = os.path.join(
            posteriors_dir, f"{sg_tag}_k{k:04d}_cycle{res.cycle_index}_posterior.csv")
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(names)
            w.writerows(samples.tolist())
        paths.append(path)
    return paths


def save_trajectories(outdir, sg_tag, k, stages):
    """Write this k's prior/BMU/BMU+EnKS trajectory ensembles + time axes as .npy."""
    trajectories_dir = os.path.join(outdir, "trajectories")
    os.makedirs(trajectories_dir, exist_ok=True)
    arrays = {
        "prior": (stages.prior_traj, stages.pc_time),
        "bmu": (stages.bmu_traj, stages.pc_time),
        "bmu_enks": (stages.bmu_enks_traj, stages.enks_time),
    }
    paths = []
    for stage, (traj, t) in arrays.items():
        traj_path = os.path.join(trajectories_dir, f"{sg_tag}_k{k:04d}_{stage}.npy")
        time_path = os.path.join(trajectories_dir, f"{sg_tag}_k{k:04d}_{stage}_time.npy")
        np.save(traj_path, traj)
        np.save(time_path, np.asarray(t))
        paths += [traj_path, time_path]
    return paths


def out_tag(sg):
    """Tag used in DATA/ and output filenames (SGA, SGB)."""
    return sg["sg_tag"]


def build_pipeline(common, sg):
    cfg = PipelineConfig(
        base_directory=common["base_directory"],
        input_dimension=common.get("input_dimension", 7),
        n_doe_samples=common.get("n_doe_samples", 1000),
        n_cycles=sg["n_cycles"],
        clogging_threshold=sg.get("threshold"),
    )
    cfg.mcmc.n_chains = common.get("n_chains", 3)
    cfg.mcmc.nb_weights = common.get("nb_weights", 1024)
    cfg.mcmc.sample_size = common.get("sample_size", 1000)
    cfg.mcmc.burn_in = common.get("burn_in", 100)
    return CloggingPipeline(cfg).setup(sg_tag=sg["sg_tag"])


def latex_sweep_table(rows_by_k, sg_display, label, threshold, unit):
    """rows_by_k: dict k -> {stage: RULStats}. Emits mean/std vs k for
    BMU and BMU+EnKS side by side."""
    def f(x):
        return "--" if (x != x) else f"{x:.1f}"
    L = [r"\begin{table}[ht]", r"  \centering",
         (rf"  \caption{{RUL mean and standard deviation (in {unit}) as a "
          rf"function of the number $k$ of ESTICOL points assimilated on the "
          rf"prognostics window, with the TVE data held fixed, for "
          rf"\textbf{{{sg_display}}} (threshold $\tau_c^\ast={threshold:g}\%$).}}"),
         rf"  \label{{{label}}}",
         r"  \begin{tabular}{rrrrr}", r"    \toprule",
         r"    & \multicolumn{2}{c}{BMU} & \multicolumn{2}{c}{BMU\,+\,EnKS}\\",
         r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"    $k$ & Mean & Std & Mean & Std \\", r"    \midrule"]
    for k in sorted(rows_by_k):
        b = rows_by_k[k]["bmu"]; e = rows_by_k[k]["bmu_enks"]
        L.append(f"    {k} & {f(b.mean)} & {f(b.std)} & {f(e.mean)} & {f(e.std)} \\\\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(L)


def run_one_sg(common, sg, outdir):
    """Run the whole obs_sizes sweep for one SG. Returns (rows_by_k, csv_rows)."""
    obs_sizes = sg["obs_sizes"]
    n_rul_samples = common.get("n_rul_samples", 1000)

    print(f"=== ESTICOL sweep: {sg['display']} ===", flush=True)
    print(f"    k values: {obs_sizes}", flush=True)
    pipe = build_pipeline(common, sg)     # earlier cycles cached on first k

    rows_by_k = {}
    csv_rows = []
    for k in obs_sizes:
        t0 = time.time()
        try:
            stats, cycle_results, stages = run_observation_size_with_stages(
                pipe, k=k,
                split_sim_hours=sg["split_sim_hours"],
                prognostics_split_hours=sg["prognostics_split_hours"],
                n_rul_samples=n_rul_samples,
            )
        except Exception as e:
            import traceback
            print(f"  k={k}: FAILED: {e}", flush=True)
            traceback.print_exc()
            continue
        posterior_paths = save_posteriors(
            outdir, out_tag(sg), k, pipe.cfg, cycle_results, n_rul_samples)
        trajectory_paths = save_trajectories(outdir, out_tag(sg), k, stages)
        for p in posterior_paths + trajectory_paths:
            print(f"    wrote {p}", flush=True)
        by_stage = {s.stage: s for s in stats}
        rows_by_k[k] = by_stage
        for st in STAGE_ORDER:
            s = by_stage[st]
            csv_rows.append(dict(sg=sg["display"], k=k, stage=st, mean=s.mean,
                                 std=s.std, cv=s.cv, q05=s.q05, median=s.median,
                                 q95=s.q95, count=s.count, n_total=s.n_total))
        dt = time.time() - t0
        print(f"  k={k:>4}: "
              f"BMU mean={by_stage['bmu'].mean:8.1f} std={by_stage['bmu'].std:7.1f} | "
              f"EnKS mean={by_stage['bmu_enks'].mean:8.1f} std={by_stage['bmu_enks'].std:7.1f} "
              f"({dt/60:.1f} min)", flush=True)

    return rows_by_k, csv_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", default="results_sweep")
    args = ap.parse_args()

    with open(args.config) as fh:
        conf = json.load(fh)
    common = conf["common"]
    sgs = conf["steam_generators"]
    os.makedirs(args.outdir, exist_ok=True)

    unit = common.get("unit", "days")
    threshold = common.get("threshold")

    all_csv_rows = []
    json_out = {}
    tables = []
    for sg in sgs:
        try:
            rows_by_k, csv_rows = run_one_sg(common, sg, args.outdir)
        except Exception as e:
            import traceback
            print(f"  {sg['display']}: FAILED: {e}", flush=True)
            traceback.print_exc()
            continue
        all_csv_rows += csv_rows
        json_out[sg["display"]] = {
            k: {st: by_stage[st].to_dict() for st in STAGE_ORDER}
            for k, by_stage in rows_by_k.items()
        }
        if rows_by_k:
            tables.append(latex_sweep_table(rows_by_k, sg["display"], sg["label"],
                                            sg.get("threshold", threshold), unit))
        print()

    # --- write outputs (csv module, no pandas dependency required) ---
    csv_path = os.path.join(args.outdir, "esticol_sweep.csv")
    cols = ["sg", "k", "stage", "mean", "std", "cv", "q05", "median", "q95",
            "count", "n_total"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader(); w.writerows(all_csv_rows)

    with open(os.path.join(args.outdir, "esticol_sweep.json"), "w") as fh:
        json.dump(json_out, fh, indent=2)

    with open(os.path.join(args.outdir, "esticol_sweep.tex"), "w") as fh:
        fh.write("\n\n".join(tables) + "\n")

    print(f"\nwrote {args.outdir}/esticol_sweep.{{csv,json,tex}}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
