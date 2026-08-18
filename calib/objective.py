"""
SEAM 3 — the objective. THE STABLE CONTRACT every method is written against.

The objective ties model + observation + data together behind one interface:

    objective(x)          -> finite scalar loss   # x is a vector in [0, 1]^d
    objective.log_likelihood(x) -> Gaussian logL
    objective.log_prior(x)      -> joint log prior (from priors.PriorSet)
    objective.log_posterior(x)  -> logL + log_prior

`free` is a list of "process.attr" names with (lo, hi) bounds; the objective
maps the unit cube to natural units, simulates, aligns to the CGM, and returns
either a loss or a Bayesian quantity off the SAME simulate-and-align core, so
they can never disagree. It never raises and never returns NaN, so any search
method can drive it.

This file deliberately knows NOTHING about optimizers or samplers. Those live
in `methods/` and consume this contract from the outside. Keep it that way:
picking or adding a method should never edit this file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

_LOG_2PI = math.log(2.0 * math.pi)
NEG_INF = float("-inf")

from .model import Protocol, simulate
from .observe import Observation
from .priors import PriorSet


@dataclass
class Param:
    name: str          # "process_id.attr"
    lo: float
    hi: float

    def to_unit(self, v):   return (v - self.lo) / (self.hi - self.lo)
    def from_unit(self, u):  return self.lo + float(np.clip(u, 0, 1)) * (self.hi - self.lo)


class Objective:
    def __init__(
        self,
        free: Sequence[Param],
        protocol: Protocol,
        cgm_t: np.ndarray,
        cgm_g: np.ndarray,
        defaults: Dict[str, float],
        warmup_s: float = 6 * 3600.0,
        project_root: str = ".",
        observation: Observation = None,
        loss: str = "rmse",
        fail_penalty: float = 1e4,
        sigma: float = 8.0,
        prior: PriorSet = None,
    ):
        self.free = list(free)
        self.protocol = protocol
        self.cgm_t = np.asarray(cgm_t, float)
        self.cgm_g = np.asarray(cgm_g, float)
        self.defaults = dict(defaults)
        self.warmup_s = warmup_s
        self.project_root = project_root
        self.obs = observation or Observation()
        self.loss_kind = loss
        self.fail_penalty = fail_penalty
        self.sigma = sigma            # CGM noise sd (mg/dL); sets likelihood scale
        self.prior = prior            # PriorSet, or None for likelihood-only
        self.duration_s = warmup_s + float(self.cgm_t[-1])
        self.history: List[dict] = []   # every evaluation, for inspection

    # -- unit-cube <-> params ------------------------------------------------
    def x0(self) -> np.ndarray:
        return np.array([p.to_unit(self.defaults[p.name]) for p in self.free])

    def overrides(self, x) -> Dict[str, float]:
        return {p.name: p.from_unit(x[i]) for i, p in enumerate(self.free)}

    # -- the shared core: simulate, align, return residuals ------------------
    def residuals(self, x):
        """pred - obs at the CGM sample times, or None if the sim failed.
        This is the one place that touches the model; loss and likelihood
        both read from here so they can never disagree."""
        x = np.asarray(x, float).ravel()
        ov = self.overrides(x)
        sim_t, sim_g = simulate(
            ov, self.protocol, self.duration_s, project_root=self.project_root
        )
        if sim_t.size < 3 or not np.isfinite(sim_g).all():
            return None, ov, None
        m = sim_t >= self.warmup_s
        st, sg = sim_t[m] - self.warmup_s, sim_g[m]
        pred, nuis = self.obs.predict(st, sg, self.cgm_t, self.cgm_g)
        return pred - self.cgm_g, ov, nuis

    # -- SEAM 3a: the LOSS (for point optimizers) ----------------------------
    def __call__(self, x) -> float:
        resid, ov, nuis = self.residuals(x)
        if resid is None:
            self.history.append({"loss": self.fail_penalty, "params": ov, "ok": False})
            return self.fail_penalty
        if self.loss_kind == "mae":
            loss = float(np.mean(np.abs(resid)))
        else:
            loss = float(np.sqrt(np.mean(resid ** 2)))
        if not np.isfinite(loss):
            loss = self.fail_penalty
        self.history.append({"loss": loss, "params": ov, "nuisance": nuis, "ok": True})
        return loss

    # -- SEAM 3b: the LOG-LIKELIHOOD (for samplers) --------------------------
    def log_likelihood(self, x) -> float:
        """Gaussian iid noise: log p(data | theta) = -SSR/(2 sigma^2) + const.
        This is the exact quantity derived from y_i = yhat_i + N(0, sigma^2).
        Failed simulations get -inf (probability zero)."""
        resid, ov, nuis = self.residuals(x)
        if resid is None:
            return NEG_INF
        n = resid.size
        ssr = float(np.sum(resid ** 2))
        return -0.5 * ssr / self.sigma ** 2 - 0.5 * n * (_LOG_2PI + 2 * math.log(self.sigma))

    def log_prior(self, x) -> float:
        """Joint log prior at x. Uniform box from the Param bounds if no
        PriorSet was supplied; otherwise the PriorSet's summed log-densities."""
        params = self.overrides(x)
        if self.prior is not None:
            return self.prior.log_prob(params)
        # default: uniform over the box -> 0 inside, -inf outside
        for p in self.free:
            v = params[p.name]
            if v < p.lo or v > p.hi:
                return NEG_INF
        return 0.0

    def log_posterior(self, x) -> float:
        """log p(theta | data) = log_likelihood + log_prior + const.
        THIS is the function MCMC and ABC drive. One line, because in log
        space Bayes' rule is just addition."""
        lp = self.log_prior(x)
        if lp == NEG_INF:            # off-support: don't even bother simulating
            return NEG_INF
        return self.log_likelihood(x) + lp

    def best(self):
        ok = [h for h in self.history if h["ok"]]
        return min(ok, key=lambda h: h["loss"]) if ok else None
