"""Construction of the KLE surrogate models used in the pipeline.

Backed by OpenTURNS Gaussian processes: ``GpMetamodelInd`` (OT KrigingAlgorithm
wrapper) + the OT ``KarhunenLoeveMetamodel``, both from ``metamodels.py``.
Mean-only. CPU-only: the GPs here are small (n ~ 700, d = 7), so OT's CPU
kriging fits comfortably fast and a GPU offers no benefit at this scale.

Two surrogate roles:

* the **base / full** surrogate, trained once on the full simulation grid and
  used to emulate whole clogging trajectories;
* the **calibration** surrogates, a small (trend x kernel) ensemble rebuilt
  inside the assimilation loop on the interpolated field-time points, each
  restricted to the single HSIC-selected active parameter via the GP's
  ``index`` argument.

On ``index`` (native OT semantics): the inner GP is *trained on the full*
``input_dimension``-D design (all parameters), so its Gram matrix is well
conditioned. ``index`` only affects *prediction* -- the other inputs are held
at their training mean and the GP is swept along the active parameter, which is
exactly what the MCMC ``CalibrationFunction`` needs when it passes a 1-D sweep
``np.asarray([x[0]])``. No wrapper/adapter is required: ``GpMetamodelInd``
implements this directly.

The trend/kernel vocabularies (``Constant/Linear/Quad``,
``AbsExp/SqExp/M-1/2/M-3/2/M-5/2``) are OT's own, so config values pass through
unchanged -- no name mapping.
"""

from __future__ import annotations

from .metamodels import GpMetamodelInd, KarhunenLoeveMetamodel


def _make_gp(trend, kernel, cfg, index=None):
    """Build one OT GP wrapper. ``nugget`` (if set on the config) maps to the
    covariance model's NuggetFactor for numerical conditioning."""
    return GpMetamodelInd(
        trend=trend,
        kernel=kernel,
        input_dimension=cfg.input_dimension,
        noise=getattr(cfg.metamodel, "nugget", None),
        index=index,
    )


def build_base_surrogate(X_train, y_train, cfg, simulation_time):
    """Fit the full mean-only KLE over the whole simulation grid (full-D GP)."""
    gp = _make_gp(cfg.metamodel.base_trend, cfg.metamodel.base_kernel, cfg, index=None)
    surrogate = KarhunenLoeveMetamodel(
        metamodel=gp,
        input_dimension=cfg.input_dimension,
        simulation_time=simulation_time,
        explained_variance_threshold=getattr(
            cfg.metamodel, "explained_variance_threshold", 0.99
        ),
        verbose=getattr(cfg, "metamodel_verbose", True),
    )
    surrogate.fit(X_train, y_train)
    return surrogate


def build_calibration_ensemble(X_train, y_train, cfg, calibration_time, active_index):
    """Build the (trend x kernel) ensemble of KLE surrogates for calibration.

    Each member's inner GP is a full-D OT kriging model with ``index`` set to
    the HSIC-selected active parameter, so MCMC can sweep that parameter at
    predict time while the others sit at their training mean.
    """
    metamodels = []
    for trend in cfg.metamodel.trends:
        for kernel in cfg.metamodel.kernels:
            gp = _make_gp(trend, kernel, cfg, index=active_index)
            member = KarhunenLoeveMetamodel(
                metamodel=gp,
                input_dimension=cfg.input_dimension,
                simulation_time=calibration_time,
                explained_variance_threshold=getattr(
                    cfg.metamodel, "explained_variance_threshold", 0.99
                ),
                verbose=False,
            )
            member.fit(X_train, y_train)
            metamodels.append(member)
    return metamodels