# Figures

`preprocessing/` holds the figures that only depend on the simulations and the field data, before any
assimilation (one per steam generator):

| file | content | produced by |
|---|---|---|
| `SGA_fig_tpd.pdf`, `SGB_fig_tpd.pdf` | THYC-Puffer-DEPO trajectory ensemble (TPD), its mean, the chemical conditioning windows, the cleanings and the prognostics start $t_P$ | `notebooks/prior_data_figures.ipynb` |
| `SGA_fig_data.pdf`, `SGB_fig_data.pdf` | the same ensemble with the ESTICOL regression points (red) and the TVE field points (blue) | `notebooks/prior_data_figures.ipynb` |
| `SGA_fig_kle_pce.pdf`, `SGB_fig_kle_pce.pdf` | Karhunen-Loeve + polynomial-chaos surrogate of the TPD and its predictivity coefficient $Q^2(t)$ | `notebooks/prior_data_figures.ipynb` |

`results/` holds the result figures of the article, as exported from the analysis notebooks:

| file | content | produced by |
|---|---|---|
| `FIG12.pdf` | density of the BMU+EnKS RUL for every ESTICOL count `k`, SG A and SG B | `notebooks/analyze_saved_results.ipynb` (`plot_rul_histograms_vs_k`) |
| `FIG13.pdf` | RUL mean and inter-quartile range vs `k`, prior / BMU / BMU+EnKS | `notebooks/analyze_saved_results.ipynb` (`plot_rul_vs_k`) |
| `FIG14.pdf` | SG B, `k=20`: RUL histograms and trajectory ensembles of the three stages with the ESTICOL (used / held out) and TVE points | `notebooks/analyze_saved_results.ipynb` (`plot_prognostics_summary`) |
| `FIGA1.pdf`, `FIGA2.pdf` | posterior samples of the seven parameters for every cycle of the full run, SG A and SG B | `notebooks/analyze_saved_results.ipynb` (`plot_posterior_grid`) |
| `FIGA3.pdf` | posterior of $a_v$ on the last cycle for every `k`, SG A and SG B | `notebooks/analyze_saved_results.ipynb` (`plot_active_posterior_vs_k`) |

These figures were produced with the published ensemble Kalman smoother (`align_observations=False`,
see the application README).

`analysis/` is where `notebooks/analyze_saved_results.ipynb` and `notebooks/exact_grid_posterior.ipynb`
write the figures of the experiments (RUL, posteriors, kinetics); it is empty in the repository.

The vertical axes of the clogging level $\tau_c$ are intentionally without values.
