"""
SMC sampler — posterior over parameters, with sigma estimated jointly.

Population of particles marched from prior to posterior through a tempering
ladder:

    p_beta(theta, sigma)  ~  prior(theta) * prior(sigma) * L(theta, sigma)^beta

beta goes 0 -> 1. At beta=0 that's just the prior (easy: draw from it). At
beta=1 it's the posterior (what you want). Each stage: reweight for a slightly
hotter target, resample to cull dead particles, then move each survivor with a
few Metropolis steps to restore diversity.

Why SMC over MCMC here: particles are independent within a stage, so it
parallelises trivially (this mini version is serial — swap the list-comp for a
process pool to scale); it seeds from the prior so there's no burn-in; and a
spread population populates multiple modes and ridges instead of one walker
missing them.

sigma (CGM noise SD) is sampled as an extra dimension with its own positive
prior (HalfNormal by default). Its posterior width is what makes every other
posterior width trustworthy — fix sigma wrong and the uncertainty is a fiction.

Output: `SMCResult` with particles (theta in natural units + sigma) and a
`.summary()` that reports each marginal against its prior, so you can SEE which
parameters the CGM actually moved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..objective import Objective
from ..priors import HalfNormal


# ------------------------------------------------------------------ helpers --
def _systematic_resample(weights: np.ndarray, rng) -> np.ndarray:
    """Low-variance resampling: one uniform, evenly spaced pointers."""
    n = weights.size
    positions = (rng.random() + np.arange(n)) / n
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0
    idx = np.zeros(n, dtype=int)
    i = j = 0
    while i < n:
        if positions[i] < cumsum[j]:
            idx[i] = j
            i += 1
        else:
            j += 1
    return idx


def _ess(logw: np.ndarray) -> float:
    """Effective sample size from unnormalised log-weights."""
    w = np.exp(logw - logw.max())
    w /= w.sum()
    return 1.0 / np.sum(w ** 2)


# ------------------------------------------------------------------- result --
@dataclass
class SMCResult:
    names: List[str]              # physiological parameter names (natural units)
    theta: np.ndarray             # (N, d) natural units
    sigma: np.ndarray             # (N,)
    logL: np.ndarray              # (N,) at beta=1
    priors: Dict                  # name -> prior object (for summary)
    sigma_prior: object
    n_sims: int = 0
    betas: List[float] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["SMC POSTERIOR  (%d particles, %d sims, tempering %s)"
                 % (self.theta.shape[0], self.n_sims,
                    " -> ".join(f"{b:.2f}" for b in self.betas)),
                 "-" * 76,
                 f"{'parameter':<44}{'post mean':>10}{'post sd':>9}{'  [5%, 95%]':>13}"]
        for i, name in enumerate(self.names):
            col = self.theta[:, i]
            lo, hi = np.percentile(col, [5, 95])
            # how much did the data move this parameter vs its prior?
            pr = self.priors.get(name)
            tag = ""
            if pr is not None and hasattr(pr, "sample"):
                rng = np.random.default_rng(0)
                prior_draws = np.array([pr.sample(rng) for _ in range(2000)])
                shrink = 1.0 - col.std() / (prior_draws.std() + 1e-12)
                if shrink > 0.5:
                    tag = "  <- data-driven (tight)"
                elif shrink < 0.1:
                    tag = "  <- prior-driven (CGM said little)"
            lines.append(f"{name:<44}{col.mean():>10.3f}{col.std():>9.3f}"
                         f"   [{lo:6.3f},{hi:6.3f}]{tag}")
        lo, hi = np.percentile(self.sigma, [5, 95])
        lines.append(f"{'sigma (CGM noise SD, mg/dL)':<44}{self.sigma.mean():>10.3f}"
                     f"{self.sigma.std():>9.3f}   [{lo:6.3f},{hi:6.3f}]")
        lines.append("")
        lines.append("'tight' marginals are identified by the CGM; 'prior-driven' ones are")
        lines.append("not — the posterior just echoes the literature prior you supplied.")
        return "\n".join(lines)


# --------------------------------------------------------------------- core --
def smc(
    obj: Objective,
    sigma_prior=None,
    n_particles: int = 64,
    ess_target: float = 0.5,      # resample/step when ESS drops below this fraction
    move_steps: int = 1,          # Metropolis moves per stage (more = better mixing, more sims)
    move_scale: float = 0.1,      # RW proposal sd in unit space
    max_stages: int = 20,
    seed: int = 0,
) -> SMCResult:
    """
    Run SMC. `obj` must have a PriorSet in `obj.prior` (Uniform/Normal from
    literature); sigma_prior defaults to HalfNormal(sd=15) — a weak positive
    prior on the CGM noise SD. Returns an SMCResult; call `.summary()`.
    """
    rng = np.random.default_rng(seed)
    d = len(obj.free)
    names = [p.name for p in obj.free]
    if obj.prior is None:
        raise ValueError("smc needs obj.prior set to a PriorSet (literature priors)")
    sigma_prior = sigma_prior or HalfNormal(sd=15.0)
    priors = obj.prior.priors
    n_sims = 0

    # --- init: draw particles from the prior ---------------------------------
    theta = np.zeros((n_particles, d))         # unit cube
    for k in range(n_particles):
        nat = obj.prior.sample(rng)            # natural units
        for i, p in enumerate(obj.free):
            theta[k, i] = np.clip(p.to_unit(nat[p.name]), 0.0, 1.0)
    sigma = np.array([sigma_prior.sample(rng) for _ in range(n_particles)])

    # cache SSR and n per particle (SSR depends only on theta, not sigma)
    ssr = np.full(n_particles, np.nan)
    nobs = obj.cgm_g.size
    for k in range(n_particles):
        r, _, _ = obj.residuals(theta[k]); n_sims += 1
        ssr[k] = np.inf if r is None else float(np.sum(r ** 2))

    def loglike(ssr_k, sig):
        if not np.isfinite(ssr_k):
            return -np.inf
        return -0.5 * ssr_k / sig ** 2 - 0.5 * nobs * (np.log(2 * np.pi) + 2 * np.log(sig))

    def logprior_th(x):
        return obj.log_prior(x)                # PriorSet + box, in unit->natural

    logL = np.array([loglike(ssr[k], sigma[k]) for k in range(n_particles)])
    logw = np.full(n_particles, -np.log(n_particles))
    beta = 0.0
    betas = [0.0]

    # --- temper prior -> posterior ------------------------------------------
    for _ in range(max_stages):
        if beta >= 1.0:
            break
        # adaptively pick next beta so ESS hits the target (bisection)
        finite = np.isfinite(logL)
        lo, hi = beta, 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            inc = (mid - beta) * np.where(finite, logL, -1e300)
            test = logw + inc
            if _ess(test) < ess_target * n_particles:
                hi = mid
            else:
                lo = mid
        beta_new = hi
        logw = logw + (beta_new - beta) * np.where(finite, logL, -1e300)
        beta = beta_new
        betas.append(round(beta, 4))

        # normalise, resample
        w = np.exp(logw - logw.max()); w /= w.sum()
        idx = _systematic_resample(w, rng)
        theta, sigma, ssr = theta[idx].copy(), sigma[idx].copy(), ssr[idx].copy()
        logL = np.array([loglike(ssr[k], sigma[k]) for k in range(n_particles)])
        logw = np.full(n_particles, -np.log(n_particles))

        # move: random-walk Metropolis on the tempered target
        for _ in range(move_steps):
            for k in range(n_particles):
                th_prop = theta[k] + rng.normal(0, move_scale, d)
                th_prop = np.clip(th_prop, 0.0, 1.0)
                sig_prop = sigma[k] * np.exp(rng.normal(0, move_scale))  # positive RW

                lp_cur = logprior_th(theta[k]) + sigma_prior.log_prob(sigma[k])
                lp_prop = logprior_th(th_prop) + sigma_prior.log_prob(sig_prop)
                if not np.isfinite(lp_prop):
                    continue

                r, _, _ = obj.residuals(th_prop); n_sims += 1
                ssr_prop = np.inf if r is None else float(np.sum(r ** 2))
                ll_cur = loglike(ssr[k], sigma[k])
                ll_prop = loglike(ssr_prop, sig_prop)

                log_accept = beta * (ll_prop - ll_cur) + (lp_prop - lp_cur)
                if np.log(rng.random() + 1e-300) < log_accept:
                    theta[k], sigma[k], ssr[k] = th_prop, sig_prop, ssr_prop
            logL = np.array([loglike(ssr[k], sigma[k]) for k in range(n_particles)])

    # convert theta back to natural units for reporting
    theta_nat = np.zeros_like(theta)
    for i, p in enumerate(obj.free):
        theta_nat[:, i] = [p.from_unit(theta[k, i]) for k in range(n_particles)]

    return SMCResult(
        names=names, theta=theta_nat, sigma=sigma, logL=logL,
        priors=priors, sigma_prior=sigma_prior, n_sims=n_sims, betas=betas,
    )
