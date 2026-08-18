#!/usr/bin/env python3
"""
End-to-end example: known meals + known CGM -> estimate parameters.

    python mini/example.py

Wires the four seams together on a tiny synthetic problem so you can see the
whole loop in one screen, then scaffold outward from here.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from calib import Objective, Observation, Param, fit, random_search, cma_es, simulate


def main():
    warmup_s = 6 * 3600.0

    # ---- KNOWN perturbation: three meals a day for 1 day --------------------
    # (t_seconds_from_sim_start, carb_g, peak_time_min)
    protocol = []
    for day in range(1):
        for hour, carbs in [(7, 45), (12, 65), (18, 75)]:
            protocol.append((warmup_s + day * 86400 + hour * 3600, carbs, 12.0))

    # ---- KNOWN CGM: generate it from the model with TRUE parameters ----------
    # (in real use you'd load this from a CSV instead)
    truth = {
        "insulin_secretion.glucose_sensitivity": 1.4,
        "glucose_uptake_muscle.insulin_sensitivity": 0.6,
    }
    duration_s = warmup_s + 1 * 86400.0
    sim_t, sim_g = simulate(truth, protocol, duration_s, project_root=ROOT)
    m = sim_t >= warmup_s
    cgm_t = sim_t[m] - warmup_s
    cgm_g = sim_g[m] + np.random.default_rng(0).normal(0, 4, m.sum())  # sensor noise
    print(f"generated CGM: {cgm_t.size} samples, "
          f"{cgm_g.min():.0f}-{cgm_g.max():.0f} mg/dL")

    # ---- estimate those two parameters back ----------------------------------
    free = [
        Param("insulin_secretion.glucose_sensitivity", 0.3, 2.0),
        Param("glucose_uptake_muscle.insulin_sensitivity", 0.2, 2.5),
    ]
    defaults = {p.name: 1.0 for p in free}   # start from the wrong (default) values

    obj = Objective(
        free=free,
        protocol=protocol,
        cgm_t=cgm_t,
        cgm_g=cgm_g,
        defaults=defaults,
        warmup_s=warmup_s,
        project_root=ROOT,
        observation=Observation(lag_min=12.0),
    )

    print(f"baseline loss (defaults): {obj(obj.x0()):.3f} mg/dL RMSE")
    print("searching...")
    #best = fit(obj, random_search(n=6, seed=1))
    # For real fits, swap in CMA-ES (needs `pip install cma`) and save results:
    #   from calib import cma_es, save_params
    best = fit(obj, cma_es(n_evals=150))
    #   save_params(best, "configs/processes.fitted.yaml")

    print("\n--- result ---")
    print(f"best loss: {best['loss']:.3f} mg/dL RMSE ({len(obj.history)} evaluations)")
    for name in defaults:
        print(f"  {name}: est {best['params'][name]:.3f}  (true {truth.get(name, '—')})")
    print(f"  observation: {best['nuisance']}")


if __name__ == "__main__":
    main()
