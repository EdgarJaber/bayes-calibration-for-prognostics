# Figures

`preprocessing/` holds the figures that only depend on the simulations and the field data, before any
assimilation (one per steam generator):

| file | content | produced by |
|---|---|---|
| `SGA_fig_tpd.pdf`, `SGB_fig_tpd.pdf` | THYC-Puffer-DEPO trajectory ensemble (TPD), its mean, the chemical conditioning windows, the cleanings and the prognostics start $t_P$ | `notebooks/prior_data_figures.ipynb` |
| `SGA_fig_data.pdf`, `SGB_fig_data.pdf` | the same ensemble with the ESTICOL regression points (red) and the TVE field points (blue) | `notebooks/prior_data_figures.ipynb` |
| `SGA_fig_kle_pce.pdf`, `SGB_fig_kle_pce.pdf` | Karhunen-Loeve + polynomial-chaos surrogate of the TPD and its predictivity coefficient $Q^2(t)$ | `notebooks/prior_data_figures.ipynb` |

`analysis/` is where `notebooks/analyze_saved_results.ipynb` and `notebooks/exact_grid_posterior.ipynb`
write the figures of the experiments (RUL, posteriors, kinetics); it is empty in the repository.

The vertical axes of the clogging level $\tau_c$ are intentionally without values.
