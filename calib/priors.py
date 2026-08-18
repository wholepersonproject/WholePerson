"""
Priors.

A prior on one parameter is nothing more than a function

    theta (natural units) -> log p(theta)

and the joint prior over all free parameters is the SUM of those (because
parameters are assumed independent, so their densities multiply, so their
LOG-densities add). That single fact is the whole module.

Everything works in LOG space:
  - a value outside a distribution's support returns -inf, which after
    exp() is probability zero. That's how a Uniform prior encodes "this
    parameter cannot be here."
  - never compute p and take the log; compute log p directly, or you lose
    all precision in the tails and -inf turns into a divide-by-zero.

Usage:

    priors = {
        "insulin_secretion.glucose_sensitivity": Normal(1.0, 0.3),
        "glucose_uptake_muscle.insulin_sensitivity": Uniform(0.2, 2.5),
        "hepatic_glucose_production.production_rate": LogNormal(2.0, 0.25),
    }
    lp = PriorSet(priors)
    lp.log_prob({"insulin_secretion.glucose_sensitivity": 1.4, ...})  # -> scalar
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

_LOG_2PI = math.log(2.0 * math.pi)
NEG_INF = float("-inf")


# =============================================================================
# One distribution = one class with a log_prob(x) method.
# Add your own by copying the pattern: return the log-density, -inf off-support.
# =============================================================================

@dataclass
class Uniform:
    """Flat inside [lo, hi], impossible outside. This is what a bare
    (lo, hi) bound already means — 'anywhere in range, no preference'."""
    lo: float
    hi: float

    def log_prob(self, x: float) -> float:
        if x < self.lo or x > self.hi:
            return NEG_INF
        return -math.log(self.hi - self.lo)

    def sample(self, rng) -> float:
        return rng.uniform(self.lo, self.hi)


@dataclass
class Normal:
    """Gaussian pull toward `mu` with spread `sd`. Use when you have a
    literature central value and a sense of how tightly you believe it."""
    mu: float
    sd: float

    def log_prob(self, x: float) -> float:
        z = (x - self.mu) / self.sd
        return -0.5 * (z * z + _LOG_2PI) - math.log(self.sd)

    def sample(self, rng) -> float:
        return rng.normal(self.mu, self.sd)


@dataclass
class LogNormal:
    """For strictly-positive parameters spanning a multiplicative range
    (rate constants, half-lives). `mu`/`sd` are on the natural-log scale:
    LogNormal(mu=log(2.0), sd=0.25) is centred on 2.0, +/- ~25% multiplicative."""
    mu: float          # mean of log(x)
    sd: float          # sd of log(x)

    def log_prob(self, x: float) -> float:
        if x <= 0:
            return NEG_INF
        z = (math.log(x) - self.mu) / self.sd
        return -0.5 * (z * z + _LOG_2PI) - math.log(self.sd) - math.log(x)

    def sample(self, rng) -> float:
        return math.exp(rng.normal(self.mu, self.sd))

    @classmethod
    def from_median_cv(cls, median: float, cv: float = 0.25) -> "LogNormal":
        """Convenience: centre on a value, spread by an approximate CV."""
        return cls(mu=math.log(median), sd=cv)


@dataclass
class HalfNormal:
    """Normal folded at zero — for strictly-positive scale parameters like a
    noise SD (sigma). This is the right prior when you'd reach for a Normal but
    the quantity can't be negative. `sd` sets the scale of plausible values."""
    sd: float

    def log_prob(self, x: float) -> float:
        if x <= 0:
            return NEG_INF
        z = x / self.sd
        return math.log(2.0) - 0.5 * (z * z + _LOG_2PI) - math.log(self.sd)

    def sample(self, rng) -> float:
        return abs(rng.normal(0.0, self.sd))


@dataclass
class TruncatedNormal:
    """Gaussian belief that also respects a hard physical bound — e.g. a
    sensitivity you think is ~1.0 but that cannot be negative."""
    mu: float
    sd: float
    lo: float = NEG_INF
    hi: float = float("inf")

    def log_prob(self, x: float) -> float:
        if x < self.lo or x > self.hi:
            return NEG_INF
        z = (x - self.mu) / self.sd
        # unnormalised over the truncation; fine for MCMC/ABC where the
        # constant cancels. Normalise with the erf CDF if you need exact density.
        return -0.5 * (z * z + _LOG_2PI) - math.log(self.sd)

    def sample(self, rng) -> float:
        for _ in range(1000):
            v = rng.normal(self.mu, self.sd)
            if self.lo <= v <= self.hi:
                return v
        return float(np.clip(self.mu, self.lo, self.hi))


# =============================================================================
# The joint prior: sum of per-parameter log-densities.
# =============================================================================

class PriorSet:
    """Independent priors over named parameters. log_prob adds; sample draws."""

    def __init__(self, priors: Dict[str, object]):
        self.priors = dict(priors)

    def log_prob(self, params: Dict[str, float]) -> float:
        """Joint log prior = sum of the per-parameter log-densities.
        Returns -inf the moment any parameter is off-support."""
        total = 0.0
        for name, dist in self.priors.items():
            lp = dist.log_prob(params[name])
            if lp == NEG_INF:
                return NEG_INF          # one impossible value kills the whole draw
            total += lp
        return total

    def sample(self, rng) -> Dict[str, float]:
        """Draw one parameter set from the prior — the natural way to seed a
        population sampler, or to sanity-check that your priors are sane."""
        return {name: dist.sample(rng) for name, dist in self.priors.items()}

    def names(self) -> Sequence[str]:
        return list(self.priors)
