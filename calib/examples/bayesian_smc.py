#!/usr/bin/env python3
"""
Full workflow: sensitivity screen -> literature priors -> SMC posterior.

    python mini/example_smc.py

This is the recommended pipeline for a single observable:
  1. screen()  decides WHICH parameters a single CGM can identify
  2. you pick the free set from the screen (tool advises, you decide)
  3. PriorSet  supplies literature priors (Uniform = range only, Normal = value+spread)
  4. smc()     returns the POSTERIOR, with sigma estimated jointly

Sizes below are tiny so it runs quickly as a demo. For a real posterior use
n_particles >= 200 and move_steps 2-5 (see the note at the bottom) — tune on
your own machine.
"""

import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from calib import (Objective, Observation, Param, simulate, screen, format_screen,
                  smc, PriorSet, Normal, Uniform, HalfNormal)


def main():
    warm = 2 * 3600.0
    dur = warm + 8 * 3600.0
    protocol = [(warm + 1 * 3600, 70, 12.0)]        # one 70 g meal

    # synthetic CGM with known truth (replace with your CSV load)
    TRUE_SIGMA = 5.0
    truth = {"insulin_secretion.glucose_sensitivity": 0.5,
             "hepatic_glucose_production.production_rate": 3.0}
    st, sg = simulate(truth, protocol, dur, project_root=ROOT)
    m = st >= warm
    cgm_t = st[m] - warm
    cgm_g = sg[m] + np.random.default_rng(0).normal(0, TRUE_SIGMA, m.sum())

    # --- 1. SCREEN (decide the free set) -------------------------------------
    candidates = [
        Param("insulin_secretion.glucose_sensitivity", 0.3, 2.0),
        Param("glucose_uptake_muscle.insulin_sensitivity", 0.2, 2.5),
        Param("hepatic_glucose_production.production_rate", 0.5, 4.0),
    ]
    obj_screen = Objective(candidates, protocol, cgm_t, cgm_g,
                           {p.name: 1.0 for p in candidates},
                           warmup_s=warm, project_root=ROOT,
                           observation=Observation(12.0))
    print(format_screen(screen(obj_screen, step=0.15)))
    print()

    # --- 2 & 3. pick free set + literature priors ----------------------------
    free = [
        Param("insulin_secretion.glucose_sensitivity", 0.3, 2.0),
        Param("hepatic_glucose_production.production_rate", 0.5, 4.0),
    ]
    prior = PriorSet({
        free[0].name: Normal(1.0, 0.5),     # believed value + spread
        free[1].name: Uniform(0.5, 4.0),    # range only, no preference
    })
    obj = Objective(free, protocol, cgm_t, cgm_g, {p.name: 1.0 for p in free},
                    warmup_s=warm, project_root=ROOT,
                    observation=Observation(12.0), prior=prior)

    # --- 4. SMC posterior (sigma estimated jointly) --------------------------
    print("true: gs=0.5, hgp=3.0, sigma=%.1f" % TRUE_SIGMA)
    res = smc(obj, sigma_prior=HalfNormal(sd=15.0),
              n_particles=64, move_steps=3, move_scale=0.1, seed=1)
    print(res.summary())


if __name__ == "__main__":
    main()
