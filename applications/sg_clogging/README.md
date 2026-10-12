# Steam-generator clogging application

Application of [`bayes_fusion`](../../bayes_fusion) to the clogging of the steam generators (SG) of
pressurized-water reactors, simulated with THYC-Puffer-DEPO. Two SGs are provided, **SG A** (tag `SGA`,
failure threshold 40 %) and **SG B** (tag `SGB`, threshold 60 %).

The industrial data are **not** distributed: put them in [`DATA/`](DATA) (git-ignored, see
[DATA/README.md](DATA/README.md)). Run everything from this folder after `pip install -e ../..`.

```
clogging_prognostics/       application package (data loading, pipeline, observation-size experiments)
run_rul_tables.py           full-data run, both SGs                    -> results/
run_esticol_sweep.py        ESTICOL observation-size (first-k) sweep   -> results_sweep/
run_esticol_kinetics.py     SG B: quasi-linear vs exponential ESTICOL sub-populations -> results_kinetics/
*_config.json               per-experiment settings (SG tags, thresholds, window anchors, MCMC)
run_*.sh, run_all_experiments.sh   detached (nohup) launchers
analyze_saved_results.ipynb        plots and tables of the three experiments
prior_data_figures.ipynb           TPD, KLE-PCE surrogate and data figures
exact_grid_posterior.ipynb         exact 1-D grid posterior (no MCMC)
likelihood_noise_study.py          effect of the surrogate-aggregation weights on the posterior
```

```bash
python run_esticol_kinetics.py --dry-run                 # list the selected ESTICOL points (fast)
./run_all_experiments.sh                                  # all three experiments, from scratch
PARALLEL=1 ARCHIVE=1 ./run_all_experiments.sh             # in parallel, moving old results aside
```

Each steam generator has its own entry (`display`, `sg_tag`, `threshold`, `n_cycles`, window anchors)
in the config files. In the ESTICOL experiments only the last cycle's ESTICOL data change (first `k`
points, or a chosen sub-population through `selection`), while the TVE field data are kept fixed.
