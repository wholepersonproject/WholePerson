#!/usr/bin/env python3
"""
Full glucose-pathway sensitivity screen.

Screens EVERY continuously-tunable parameter on the glucose pathway (72 of
them) against a CGM, in one run. Reports:
  - each parameter ranked by how much it moves the CGM (informative / weak / flat)
  - degenerate pairs (parameters that move the trace the same way)

Purpose: turn "there are a lot of params" into a data-backed free set. Free the
'informative' ones, fix the 'flat' ones, keep only one from each degenerate
pair. THAT subset is what a single CGM can actually identify.

Run from the repo root:
    python mini/screen_all.py

Cost: ~2 simulations per parameter (~145 sims). On a short window that's a few
minutes; lengthen `dur`/add meals for a more realistic screen once you've seen
it work. Many of the second-order knobs (reference concentrations, half-maxes,
oscillator internals) will screen flat against one CGM — that's the expected,
useful result, not a bug.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from calib import Objective, Observation, Param, simulate, screen, format_screen


# ============================================================================
# ALL glucose-pathway parameters (name, lo, hi) — pulled from processes.yaml
# metadata ranges. Comment out any you want to skip.
# ============================================================================
CANDIDATES = [
    # --- insulin secretion (beta cell) ---
    Param("insulin_secretion.glucose_sensitivity", 0.3, 2.0),
    Param("insulin_secretion.basal_insulin", 0.2, 3.0),
    Param("insulin_secretion.max_insulin", 50, 300),
    Param("insulin_secretion.clearance_per_min", 0.01, 1.0),
    Param("insulin_secretion.glucose_K", 100, 250),
    Param("insulin_secretion.glucose_n", 1.5, 8.0),
    Param("insulin_secretion.fed_multiplier", 1.0, 2.0),
    Param("insulin_secretion.fasted_multiplier", 0.4, 1.0),
    Param("insulin_secretion.sst_ref", 0, 30),
    Param("insulin_secretion.sst_k", 1, 200),
    Param("insulin_secretion.pp_ref", 0, 1000),
    Param("insulin_secretion.pp_k", 1, 2000),
    Param("insulin_secretion.glp1_ref", 0, 20),
    Param("insulin_secretion.glp1_k", 15, 100),
    Param("insulin_secretion.glp1_amp", 0, 2.0),
    Param("insulin_secretion.osc_tau_a", 0.1, 2.0),
    Param("insulin_secretion.osc_tau_r", 1, 20),
    Param("insulin_secretion.osc_gamma", 0.2, 2.0),
    Param("insulin_secretion.osc_beta0", -1.0, 1.0),
    Param("insulin_secretion.osc_beta1", 0, 2.0),
    Param("insulin_secretion.osc_K", 80, 200),
    Param("insulin_secretion.osc_n", 1, 8),
    Param("insulin_secretion.pulse_frac", 0, 1.0),
    # --- glucagon secretion (alpha cell / counter-regulation) ---
    Param("glucagon_secretion.basal_glucagon", 60, 300),
    Param("glucagon_secretion.clearance_per_min", 0.01, 1.0),
    Param("glucagon_secretion.fasted_multiplier", 1.0, 2.0),
    Param("glucagon_secretion.fed_multiplier", 0.4, 1.0),
    Param("glucagon_secretion.pp_ref", 0, 150),
    Param("glucagon_secretion.pp_k", 150, 1000),
    Param("glucagon_secretion.sst_ref", 0, 30),
    Param("glucagon_secretion.sst_k", 10, 60),
    Param("glucagon_secretion.amylin_ref", 0, 15),
    Param("glucagon_secretion.amylin_k", 20, 150),
    Param("glucagon_secretion.glp1_ref", 0, 20),
    Param("glucagon_secretion.glp1_k", 15, 100),
    Param("glucagon_secretion.insulin_k", 10, 100),
    # --- remote insulin (action delay) ---
    Param("remote_insulin.action_td_min", 3, 30),
    # (remote_insulin.n_stages is an integer count -> not screened)
    # --- hepatic glucose production (liver) ---
    Param("hepatic_glucose_production.production_rate", 1.5, 3.0),
    Param("hepatic_glucose_production.glucose_ref", 60, 120),
    Param("hepatic_glucose_production.glucose_k", 30, 150),
    Param("hepatic_glucose_production.cortisol_ref", 0, 25),
    Param("hepatic_glucose_production.cortisol_k", 5, 50),
    Param("hepatic_glucose_production.cortisol_amp", 0, 2.0),
    # --- tissue glucose uptake (insulin-dependent) ---
    Param("glucose_uptake_muscle.basal_rate", 0.005, 0.5),
    Param("glucose_uptake_muscle.insulin_sensitivity", 0.2, 2.5),
    Param("glucose_uptake_muscle.adiponectin_ref", 0, 30),
    Param("glucose_uptake_muscle.adiponectin_k", 5, 50),
    Param("glucose_uptake_muscle.adiponectin_amp", 0, 1.5),
    Param("glucose_uptake_muscle.resistin_ref", 0, 30),
    Param("glucose_uptake_muscle.resistin_k", 10, 80),
    Param("glucose_uptake_heart.basal_rate", 0.005, 0.5),
    Param("glucose_uptake_heart.insulin_sensitivity", 0.2, 2.5),
    Param("glucose_uptake_heart.adiponectin_ref", 0, 30),
    Param("glucose_uptake_heart.adiponectin_k", 5, 50),
    Param("glucose_uptake_heart.adiponectin_amp", 0, 1.5),
    Param("glucose_uptake_heart.resistin_ref", 0, 30),
    Param("glucose_uptake_heart.resistin_k", 10, 80),
    Param("glucose_uptake_adipose.basal_rate", 0.005, 0.5),
    Param("glucose_uptake_adipose.insulin_sensitivity", 0.2, 2.5),
    Param("glucose_uptake_adipose.adiponectin_ref", 0, 30),
    Param("glucose_uptake_adipose.adiponectin_k", 5, 50),
    Param("glucose_uptake_adipose.adiponectin_amp", 0, 1.5),
    Param("glucose_uptake_adipose.resistin_ref", 0, 30),
    Param("glucose_uptake_adipose.resistin_k", 10, 80),
    # --- gastric emptying + intestinal absorption (meal timing) ---
    Param("gastric_emptying.max_emptying_rate", 0.3, 2.0),
    Param("gastric_emptying.half_saturation_g", 1, 60),
    Param("gastric_emptying.glp1_ref", 0, 20),
    Param("gastric_emptying.glp1_k", 10, 100),
    Param("gastric_emptying.amylin_ref", 0, 15),
    Param("gastric_emptying.amylin_k", 15, 120),
    Param("intestinal_glucose_absorption.half_absorption_min", 5, 90),
    Param("intestinal_glucose_absorption.blood_volume_dL", 30, 70),
]


def main():
    # --- CGM to screen against ------------------------------------------------
    # Synthetic here so it runs out of the box. Replace this block with your
    # real CGM arrays (cgm_t seconds-from-start, cgm_g mg/dL) and your meal log.
    warm = 2 * 3600.0
    dur = warm + 10 * 3600.0
    protocol = [(warm + 1 * 3600, 60, 12.0), (warm + 5 * 3600, 75, 12.0)]

    truth = {
        "insulin_secretion.glucose_sensitivity": 0.6,
        "glucose_uptake_muscle.insulin_sensitivity": 0.5,
        "hepatic_glucose_production.production_rate": 2.6,
    }
    st, sg = simulate(truth, protocol, dur, project_root=ROOT)
    m = st >= warm
    cgm_t = st[m] - warm
    cgm_g = sg[m] + np.random.default_rng(0).normal(0, 4, m.sum())

    print(f"screening {len(CANDIDATES)} parameters against a "
          f"{cgm_t[-1]/3600:.0f}h CGM ({cgm_t.size} samples)...")
    print(f"~{2*len(CANDIDATES)} simulations; this will take a few minutes.\n")

    obj = Objective(CANDIDATES, protocol, cgm_t, cgm_g,
                    {p.name: 1.0 for p in CANDIDATES},
                    warmup_s=warm, project_root=ROOT,
                    observation=Observation(lag_min=12.0))

    result = screen(obj, step=0.15)
    print(format_screen(result))


if __name__ == "__main__":
    main()