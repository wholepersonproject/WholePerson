"""
calib — a minimal, modular parameter-fitting scaffold for PhysiomeTwin.

The CORE is four files and one contract. Read these to understand everything:

    model.simulate       params + protocol -> glucose trace      (SEAM 1)
    observe.Observation  plasma glucose    -> predicted CGM      (SEAM 2)
    objective.Objective  x in [0,1]^d      -> loss / logL / logP (SEAM 3)
    priors.PriorSet      params            -> joint log prior

The METHOD is whatever you choose to drive that objective, and it lives in the
`methods/` sub-package so the core never has to change when you pick or add one:

    methods.optimize     fit, random_search, cma_es, scipy_local  (SEAM 4)
    methods.sample       smc, SMCResult
    methods.screen       screen, format_screen   (run FIRST — what's identifiable)
    methods.validate     validate, compare, format_report  (run LAST — how good)

The EDGES (disk I/O) are in `data.py`: load_cgm, save_params.

Minimal on purpose: no cache, no parallelism, no QC, no metric zoo. Add those
in the one file that needs them. See examples/ for full runs.
"""

# --- core: the stable contract ------------------------------------------------
from .model import simulate, Protocol
from .observe import Observation
from .objective import Objective, Param
from .priors import (PriorSet, Uniform, Normal, LogNormal, TruncatedNormal,
                     HalfNormal)

# --- methods: the swappable annex (re-exported flat for convenience) ----------
from .methods import (fit, random_search, cma_es, scipy_local,
                      smc, SMCResult, screen, format_screen,
                      validate, compare, format_report)

# --- edges: disk I/O ----------------------------------------------------------
from .data import load_cgm, save_params

__all__ = [
    # core
    "simulate", "Protocol", "Observation", "Objective", "Param",
    "PriorSet", "Uniform", "Normal", "LogNormal", "TruncatedNormal", "HalfNormal",
    # methods
    "fit", "random_search", "cma_es", "scipy_local",
    "smc", "SMCResult", "screen", "format_screen",
    "validate", "compare", "format_report",
    # edges
    "load_cgm", "save_params",
]
