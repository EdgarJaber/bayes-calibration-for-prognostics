#!/usr/bin/env python3
"""Run one full pass of the hybrid methodology for each steam generator and
write the two RUL-statistics LaTeX tables (Prior / BMU / BMU+EnKS).

Config is read from a JSON file (default configs/rul_tables_config.json). Run it from the
application root (applications/sg_clogging). Usage:

    python scripts/full_run.py --config configs/rul_tables_config.json --outdir results

Outputs, in --outdir:
    rul_tables.tex               both LaTeX tables (paste-ready)
    rul_stats.json                the raw statistics per SG/stage (for the record)
    posteriors/<SG>_cycle<i>_posterior.csv   posterior parameter samples, one file per window
    trajectories/<SG>_<stage>.npy            prior/bmu/bmu_enks trajectory ensembles
    trajectories/<SG>_<stage>_time.npy       matching time axis (days) for each stage
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
APP = os.path.abspath(os.path.join(HERE, '..'))                       # applications/sg_clogging
sys.path.insert(0, APP)                                               # clogging_prognostics
sys.path.insert(0, os.path.abspath(os.path.join(APP, '..', '..')))    # bayes_fusion (repo root)

from clogging_prognostics import (CloggingPipeline, PipelineConfig, load_thresholds,
                                  run_observation_size_with_stages, threshold_symbol)

STAGE_ORDER = ["prior", "bmu", "bmu_enks"]
STAGE_LABEL = {"prior": "Prior", "bmu": "BMU", "bmu_enks": r"BMU\,+\,EnKS"}


def save_posteriors(outdir, sg_tag, cfg, cycle_results, n_samples):
    """Write each cycle's posterior parameter samples to its own CSV.

    Only the HSIC-active variable actually moves away from its prior in a
    given cycle; the rest stay at that cycle's prior marginal -- all
    ``input_dimension`` columns are written regardless, so the file is
    self-describing. Plain csv module (no pandas dependency).
    """
    posteriors_dir = os.path.join(outdir, "posteriors")
    os.makedirs(posteriors_dir, exist_ok=True)
    names = list(cfg.prior.names)
    paths = []
    for res in cycle_results:
        samples = np.asarray([
            np.asarray(m.getSample(n_samples)).reshape(-1) for m in res.marginals
        ]).T                                    # shape (n_samples, input_dimension)
        path = os.path.join(posteriors_dir, f"{sg_tag}_cycle{res.cycle_index}_posterior.csv")
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(names)
            w.writerows(samples.tolist())
        paths.append(path)
    return paths


def save_trajectories(outdir, sg_tag, stages):
    """Write the prior/BMU/BMU+EnKS trajectory ensembles + time axes as .npy."""
    trajectories_dir = os.path.join(outdir, "trajectories")
    os.makedirs(trajectories_dir, exist_ok=True)
    arrays = {
        "prior": (stages.prior_traj, stages.pc_time),
        "bmu": (stages.bmu_traj, stages.pc_time),
        "bmu_enks": (stages.bmu_enks_traj, stages.enks_time),
    }
    paths = []
    for stage, (traj, t) in arrays.items():
        traj_path = os.path.join(trajectories_dir, f"{sg_tag}_{stage}.npy")
        time_path = os.path.join(trajectories_dir, f"{sg_tag}_{stage}_time.npy")
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
        clogging_threshold=load_thresholds(common["base_directory"])[sg["sg_tag"]],
    )
    # MCMC settings (real-run values, not the smoke-test n_chains=1)
    cfg.mcmc.n_chains = common.get("n_chains", 3)
    cfg.mcmc.nb_weights = common.get("nb_weights", 1024)
    cfg.mcmc.sample_size = common.get("sample_size", 1000)
    cfg.mcmc.burn_in = common.get("burn_in", 100)
    return CloggingPipeline(cfg).setup(sg_tag=sg["sg_tag"])


def run_one(common, sg, outdir):
    pipe = build_pipeline(common, sg)
    n_rul_samples = common.get("n_rul_samples", 1000)
    stats, cycle_results, stages = run_observation_size_with_stages(
        pipe,
        k=common.get("k_all", 10_000),           # keep ALL ESTICOL (full run)
        split_sim_hours=sg["split_sim_hours"],
        prognostics_split_hours=sg["prognostics_split_hours"],
        n_rul_samples=n_rul_samples,
    )
    sg_tag = out_tag(sg)
    posterior_paths = save_posteriors(outdir, sg_tag, pipe.cfg, cycle_results, n_rul_samples)
    trajectory_paths = save_trajectories(outdir, sg_tag, stages)
    for p in posterior_paths + trajectory_paths:
        print(f"  wrote {p}", flush=True)
    return {s.stage: s for s in stats}


def latex_table(by_stage, sg_display, label, symbol, unit, dp=1):
    def f(x):
        return "--" if (x != x) else f"{x:.{dp}f}"   # NaN -> --
    L = [r"\begin{table}[ht]", r"  \centering",
         (rf"  \caption{{RUL statistics on the prognostics window for a "
          rf"specified threshold ${symbol}$, at each stage "
          rf"of the methodology for \textbf{{{sg_display}}}. Values in {unit}.}}"),
         rf"  \label{{{label}}}",
         r"  \begin{tabular}{lrrrrr}", r"    \toprule",
         r"    Stage & Mean & Std & $q_{05}$ & Median & $q_{95}$ \\",
         r"    \midrule"]
    for st in STAGE_ORDER:
        s = by_stage[st]
        L.append(f"    {STAGE_LABEL[st]} & {f(s.mean)} & {f(s.std)} & "
                 f"{f(s.q05)} & {f(s.median)} & {f(s.q95)} \\\\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=os.path.join(APP, "configs", "rul_tables_config.json"))
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    with open(args.config) as fh:
        conf = json.load(fh)
    common = conf["common"]
    sgs = conf["steam_generators"]
    os.makedirs(args.outdir, exist_ok=True)

    unit = common.get("unit", "days")

    all_stats = {}
    tables = []
    for sg in sgs:
        name = sg["display"]
        print(f"=== {name} ===", flush=True)
        t0 = time.time()
        try:
            by_stage = run_one(common, sg, args.outdir)
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}", flush=True)
            traceback.print_exc()
            continue
        dt = time.time() - t0
        for st in STAGE_ORDER:
            s = by_stage[st]
            print(f"  {st:9s} mean={s.mean:9.1f} std={s.std:9.1f} "
                  f"median={s.median:9.1f} cross={s.count}/{s.n_total}", flush=True)
        print(f"  done in {dt/60:.1f} min", flush=True)

        all_stats[name] = {st: by_stage[st].to_dict() for st in STAGE_ORDER}
        tables.append(latex_table(by_stage, name, sg["label"], threshold_symbol(sg["sg_tag"]), unit))

    with open(os.path.join(args.outdir, "rul_tables.tex"), "w") as fh:
        fh.write("\n\n".join(tables) + "\n")
    with open(os.path.join(args.outdir, "rul_stats.json"), "w") as fh:
        json.dump(all_stats, fh, indent=2)

    print(f"\nwrote {args.outdir}/rul_tables.tex and rul_stats.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
