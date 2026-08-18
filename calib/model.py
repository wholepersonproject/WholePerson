"""
SEAM 1 — the model.

    simulate(param_overrides, protocol) -> (t_seconds, glucose_mgdl)

This is the ONLY file that knows anything about PhysiomeTwin internals. Swap
its body to point at a different simulator and nothing else in the scaffold
changes.

Two things it must get right, both of which the repo's original calibrator got
wrong:
  1. Build the model FRESH every call. Several processes carry hidden state
     across steps, so reusing an engine makes the loss depend on call order.
  2. Inject parameters at construction, and fire the protocol's meals on the
     simulation clock (the engine's own run() can't schedule mid-run).
"""

from __future__ import annotations

import contextlib
import io
from importlib import import_module
from typing import Dict, List, Tuple

import numpy as np
import yaml

# --- protocol: a plain list of meals -----------------------------------------
# A protocol is just [(t_seconds, carb_grams, peak_time_min), ...].
# Keep it dumb; richer scheduling is a later concern.
Protocol = List[Tuple[float, float, float]]


def simulate(
    param_overrides: Dict[str, float],
    protocol: Protocol,
    duration_s: float,
    project_root: str = ".",
    dt: float = 60.0,
    record_every_s: float = 300.0,
    signal=("blood", "glucose"),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run one simulation. `param_overrides` keys are "process_id.attr".
    Returns (t, glucose) sampled every `record_every_s`. NaNs on failure.
    """
    import os, sys
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from core.state import SimulationState
    from core.entity_factory import EntityFactory
    from core.perturbation import PerturbationManager
    from engine.engine import PhysiologyEngine

    # nest overrides: "insulin_secretion.glucose_sensitivity" -> {proc: {attr: v}}
    nested: Dict[str, Dict[str, float]] = {}
    for name, value in param_overrides.items():
        proc, attr = name.split(".", 1)
        nested.setdefault(proc, {})[attr] = float(value)

    cfg = os.path.join(project_root, "configs")
    with open(os.path.join(cfg, "processes.yaml")) as f:
        registry = yaml.safe_load(f)["processes"]

    t_list, g_list = [], []
    tgt, sig = signal

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            state = SimulationState()
            state.enforce_constraints = False
            EntityFactory(os.path.join(cfg, "anatomy.yaml")).initialize_simulation_state(state)
            engine = PhysiologyEngine(state)

            # build every process, folding overrides into its constructor args
            for pid, pcfg in registry.items():
                mod, cls = pcfg["class"].rsplit(".", 1)
                model_cls = getattr(import_module(mod), cls)
                params = dict(pcfg.get("parameters", {}) or {})
                params.update(nested.get(pid, {}))
                engine.register_model(pid, model_cls(**params), pcfg.get("dependencies", []))

            pm = PerturbationManager(os.path.join(cfg, "perturbations.yaml"))
            engine.set_perturbation_manager(pm)

            meals = sorted(protocol)
            i, next_rec = 0, 0.0
            while state.time < duration_s:
                while i < len(meals) and meals[i][0] <= state.time:
                    _, carbs, ptime = meals[i]
                    pm.add_perturbation("dietary", "mixed_meal",
                                        start_time=state.time,
                                        carb_grams=carbs, peak_time=ptime)
                    i += 1
                if state.time >= next_rec:
                    t_list.append(state.time)
                    v = state.get_signal(tgt, sig)
                    g_list.append(np.nan if v is None else float(v))
                    next_rec += record_every_s
                engine.step(dt)
    except Exception:
        return np.array([]), np.array([])

    return np.asarray(t_list, float), np.asarray(g_list, float)
