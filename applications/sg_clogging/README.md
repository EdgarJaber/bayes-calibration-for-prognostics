# Steam-generator clogging application

Application of [`bayes_fusion`](../../bayes_fusion) to the clogging of the steam generators (SG) of
pressurized-water reactors, simulated with THYC-Puffer-DEPO. Two SGs are provided, **SG A** (tag `SGA`,
failure threshold 40 %) and **SG B** (tag `SGB`, threshold 60 %).

The industrial data are **not** distributed: put them in [`DATA/`](DATA) (git-ignored, see
[DATA/README.md](DATA/README.md)). Install the package once with `pip install -e ../..`, then run
everything **from this folder**: `DATA/`, `configs/` and `results*/` are relative to it.

## Layout

```
clogging_prognostics/   the application package: data loading, pipeline, experiments, EnKS over cycles
configs/                one JSON per experiment: SG tags, thresholds, window anchors, MCMC settings
scripts/                the three experiments (python entry points)
run.sh                  detached launcher for the scripts (see below)
notebooks/              analysis and figure notebooks
studies/                methodological studies (likelihood weights, EnKS alignment)
figures/                preprocessing figures (TPD, KLE-PCE surrogate, data) -- see figures/README.md
DATA/                   the confidential data (not tracked)
```

## Experiments

| experiment | script | config | output |
|---|---|---|---|
| full-data run, both SGs | `scripts/full_run.py` | `configs/rul_tables_config.json` | `results/` |
| ESTICOL first-`k` sweep, both SGs | `scripts/esticol_sweep.py` | `configs/esticol_sweep_config.json` | `results_sweep/` |
| SG B kinetics (quasi-linear vs exponential ESTICOL sub-populations) | `scripts/esticol_kinetics.py` | `configs/esticol_kinetics_config.json` | `results_kinetics/` |

```bash
./run.sh full                        # one experiment, detached; logs in <outdir>/<name>_<timestamp>.log
./run.sh kinetics --only exponential # extra arguments go to the script
./run.sh all                         # the three, one after the other (PARALLEL=1: at once, ARCHIVE=1: keep old results)
python scripts/esticol_kinetics.py --dry-run   # list the selected ESTICOL points (fast, no assimilation)
```

In the ESTICOL experiments only the last cycle's ESTICOL data change (the first `k` points, or a
chosen sub-population through `selection`); the TVE field data are kept fixed.

## Notebooks

| notebook | content |
|---|---|
| `analyze_saved_results.ipynb` | trajectories, RUL statistics and tables, posteriors of the three experiments |
| `prior_data_figures.ipynb` | TPD, KLE-PCE surrogate with its predictivity, and the data figures |
| `exact_grid_posterior.ipynb` | exact 1-D grid posterior of the active parameter, without MCMC |
| `two_sg_fusion_methodology.ipynb`, `rul_observation_sweep.ipynb`, `visualize_clogging_data.ipynb` | earlier exploratory versions |

Notebooks save their figures under `figures/preprocessing/` and `figures/analysis/`.

## Studies

- `studies/likelihood_noise_study.py`: effect of the surrogate-aggregation weights (random vs fixed
  Sobol design, max vs log-sum-exp) on the posterior.
- `studies/enks_alignment_study.py`: effect of the observation alignment in the ensemble Kalman
  smoother, see below.

## Known issue: EnKS observation alignment

`smooth_cycles(..., align_observations=False)` -- the default, which reproduces the published numbers --
uses the rank of each observation inside its channel as the *column* of the trajectory ensemble, so
the i-th observation updates the i-th column of the cycle's time axis instead of the column holding
its own time (and the axis only contains the first channel's observation times).
`align_observations=True` assimilates every observation at its own time.
`studies/enks_alignment_study.py` quantifies the difference on the saved results.
