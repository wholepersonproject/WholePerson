#!/usr/bin/env python3
"""
Fit on one window, then VALIDATE on a HELD-OUT window.

    python calib/examples/validate.py

Fitting RMSE always looks good — the optimizer minimised it. The honest test is
whether the fitted parameters still track glucose on data they were NOT fit to.
This fits two parameters on a breakfast+lunch window, then reports baseline
(defaults) vs fitted metrics on a separate dinner window the fit never saw.

The trick is just: build a SECOND Objective on the held-out CGM+protocol with
the same `free` list, and hand it the fitted params. Same objective contract,
different data.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from calib import (Objective, Observation, Param, fit, cma_es, simulate,
                   compare, format_report)


def make_cgm(protocol, truth, warm, dur, seed):
    """Synthetic CGM from known truth (stand-in for a real CSV load)."""
    st, sg = simulate(truth, protocol, dur, project_root=ROOT)
    m = st >= warm
    return st[m] - warm, sg[m] + np.random.default_rng(seed).normal(0, 4, m.sum())


def main():
    warm = 6 * 3600.0
    truth = {"insulin_secretion.glucose_sensitivity": 1.4,
             "glucose_uptake_muscle.insulin_sensitivity": 0.6}
    free = [Param("insulin_secretion.glucose_sensitivity", 0.3, 2.0),
            Param("glucose_uptake_muscle.insulin_sensitivity", 0.2, 2.5)]
    defaults = {p.name: 1.0 for p in free}
    obs = Observation(lag_min=12.0)

    # ---- FIT window: breakfast + lunch --------------------------------------
    fit_protocol = [(warm + 7 * 3600, 45, 12.0), (warm + 12 * 3600, 65, 12.0)]
    ft, fg = make_cgm(fit_protocol, truth, warm, warm + 16 * 3600.0, seed=0)
    obj_fit = Objective(free, fit_protocol, ft, fg, defaults,
                        warmup_s=warm, project_root=ROOT, observation=obs)

    print("fitting on breakfast+lunch window...")
    best = fit(obj_fit, cma_es(n_evals=120))
    print("fitted:", {k: round(v, 3) for k, v in best["params"].items()},
          f"(true: gs=1.4, is=0.6)\n")

    # ---- HELD-OUT window: a dinner the fit never saw ------------------------
    ho_protocol = [(warm + 18 * 3600, 75, 12.0)]
    ht, hg = make_cgm(ho_protocol, truth, warm, warm + 24 * 3600.0, seed=1)
    obj_ho = Objective(free, ho_protocol, ht, hg, defaults,
                       warmup_s=warm, project_root=ROOT, observation=obs)

    # baseline (defaults) vs fitted, ON THE HELD-OUT DATA
    print(format_report(compare(obj_ho, best["params"])))


if __name__ == "__main__":
    main()
