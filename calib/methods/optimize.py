"""
SEAM 4 (point estimate) — optimizers that drive the objective's LOSS.

`fit(objective, driver)` runs any `driver(f, x0, bounds)` against the loss and
returns the best evaluation from the objective's own history — so a driver
can't lie about where the optimum is. A driver is just a callable that calls
`f(x)` however it likes; that is the entire contract.

Add your own optimizer as one more function here that returns a `driver`; it
never needs to touch objective.py. To add a whole different method FAMILY
(a sampler, ABC, etc.), add a sibling module in this package instead — see
`sample.py`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..objective import Objective


def fit(objective: Objective, driver: Callable) -> dict:
    """
    driver(f, x0, bounds) drives the search. Its return value is ignored;
    the best evaluation is taken from the objective's own history, so a driver
    can't lie about where the optimum is.
    """
    d = len(objective.free)
    driver(objective, objective.x0(), [(0.0, 1.0)] * d)
    return objective.best()


def random_search(n=100, seed=0):
    """Toy driver — swap for scipy/CMA-ES/etc. Here only to prove the seam."""
    def driver(f, x0, bounds):
        rng = np.random.default_rng(seed)
        f(x0)                                   # always score the defaults
        for _ in range(n):
            f(rng.random(len(x0)))
    return driver


def cma_es(n_evals=200, sigma0=0.25, seed=0):
    """
    Real derivative-free optimizer (CMA-ES). Drop-in replacement for
    random_search. Needs `pip install cma`. Searches the unit cube; the
    Objective handles the mapping to natural units.

    This is the recommended default for this model: derivative-free (so it
    tolerates the non-smooth, occasionally-failing simulator), and it scales
    fine to the handful of parameters CGM can actually identify.
    """
    def driver(f, x0, bounds):
        import cma
        if len(x0) < 2:
            raise ValueError(
                "CMA-ES needs >= 2 free parameters; for 1-D use scipy_local() "
                "or random_search()."
            )
        f(x0)                                   # score defaults first
        es = cma.CMAEvolutionStrategy(
            list(np.clip(x0, 0.01, 0.99)), sigma0,
            {"bounds": [0.0, 1.0], "seed": seed,
             "maxfevals": n_evals, "verbose": -9},
        )
        while not es.stop():
            xs = es.ask()
            es.tell(xs, [f(x) for x in xs])
        return es.result.xbest
    return driver


def scipy_local(method="Nelder-Mead", maxiter=200):
    """Local derivative-free polish, e.g. to refine a CMA-ES result.
    Uses only stdlib scipy."""
    def driver(f, x0, bounds):
        from scipy.optimize import minimize
        f(x0)
        minimize(f, np.asarray(x0, float), method=method,
                 bounds=bounds if method in ("L-BFGS-B", "Powell") else None,
                 options={"maxiter": maxiter})
    return driver
