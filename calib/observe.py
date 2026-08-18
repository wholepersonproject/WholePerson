"""
SEAM 2 — the observation model.

    predict(sim_t, sim_g, obs_t) -> glucose predicted at the CGM sample times

A CGM is not blood glucose: it lags (interstitial fluid) and has a calibration
gain/offset. The default handles both. Swap this class for anything with the
same `predict` signature — a pure interpolator, a two-compartment sensor, a
learned model.

`lag_min=0` and identity gain/offset gives you the trivial "just interpolate"
version to start from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def _interstitial(t, g, tau_s):
    """First-order plasma -> interstitial transfer (exact for linear input)."""
    if tau_s <= 0 or t.size < 2:
        return np.asarray(g, float).copy()
    out = np.empty_like(g, dtype=float)
    out[0] = g[0]
    h = np.diff(t)
    a = np.exp(-h / tau_s)
    for k in range(t.size - 1):
        out[k + 1] = a[k] * out[k] + (1 - a[k]) * g[k + 1]
    return out


@dataclass
class Observation:
    """Thin sensor layer: lag + noise + (optionally) a TINY calibration nudge.

    The sensor's job is to turn true plasma glucose into what the CGM reads —
    a small blur (interstitial lag) plus jitter, and at most a near-identity
    calibration. It must NOT reshape the signal; all shaping (level, spikes,
    timing) has to come from the model, because those are the parameters you're
    learning. So the affine correction is OFF by default, and when on it is
    clamped to a real device's calibration range (gain ~1, small offset). If
    the fit wants gain or offset outside those clamps, that's the model failing
    to produce the signal, and you want to SEE that in the loss, not launder it.
    """
    lag_min: float = 12.0
    gain: float = 1.0
    offset: float = 0.0
    fit_affine: bool = False              # OFF by default: model must make the signal
    gain_bounds: Tuple[float, float] = (0.9, 1.1)     # +/-10%, real calibration only
    offset_bounds: Tuple[float, float] = (-5.0, 5.0)  # mg/dL, real calibration only

    def predict(self, sim_t, sim_g, obs_t, obs_g=None) -> Tuple[np.ndarray, dict]:
        lagged = _interstitial(np.asarray(sim_t, float), np.asarray(sim_g, float),
                               self.lag_min * 60.0)
        yhat = np.interp(obs_t, sim_t, lagged)

        gain, offset = self.gain, self.offset
        if self.fit_affine and obs_g is not None and yhat.size >= 3:
            # weighted linear least squares: obs ~ gain*yhat + offset,
            # then CLAMP to calibration-only ranges. A clamp that bites is a
            # signal the model (not the sensor) is wrong.
            A = np.vstack([yhat, np.ones_like(yhat)]).T
            g, o = np.linalg.lstsq(A, np.asarray(obs_g, float), rcond=None)[0]
            gain = float(np.clip(g, *self.gain_bounds))
            offset = float(np.clip(o, *self.offset_bounds))

        return gain * yhat + offset, {"lag_min": self.lag_min, "gain": float(gain), "offset": float(offset)}
