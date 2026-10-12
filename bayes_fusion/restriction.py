"""Restriction of a multi-input model to one calibrated parameter."""

from __future__ import annotations

import numpy as np


class RestrictedModel:
    """Evaluate ``model`` along one input, all the others frozen at ``ref_point``.

    This is the object the calibration step needs: the MCMC sweeps a single (active)
    parameter while the remaining inputs sit at a reference value (the prior mean, say).

    Parameters
    ----------
    model : callable or object with ``.predict``
        Maps an array of shape (n, d) to an array of shape (n, T). OpenTURNS functions,
        scikit-learn-style surrogates and plain Python callables are all accepted.
    index : int
        Position of the free (active) input.
    ref_point : array-like, shape (d,)
        Values of the frozen inputs (the entry at ``index`` is ignored).
    """

    def __init__(self, model, index: int, ref_point):
        self.model = model
        self.index = int(index)
        self.ref_point = np.asarray(ref_point, dtype=float).reshape(-1)
        self.input_dimension = 1

    def predict(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        X = np.tile(self.ref_point, (x.size, 1))
        X[:, self.index] = x
        f = self.model.predict if hasattr(self.model, "predict") else self.model
        return np.asarray(f(X), dtype=float).reshape(x.size, -1)
