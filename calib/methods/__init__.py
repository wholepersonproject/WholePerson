"""
methods — the swappable annex. Everything method-SPECIFIC lives here; the
package core (model / observe / objective / priors) knows nothing about it.

    optimize.py   point estimate   fit, random_search, cma_es, scipy_local
    sample.py     Bayesian         smc, SMCResult
    screen.py     identifiability  screen, format_screen   (run this FIRST)
    validate.py   post-fit check   validate, compare, format_report

Undecided which method to use? That's fine — the core doesn't care. When you
pick one, you import it from here. When you want a new one, you add ONE file
here that consumes `objective.Objective`, and nothing else changes.
"""

from .optimize import fit, random_search, cma_es, scipy_local
from .sample import smc, SMCResult
from .screen import screen, format_screen
from .validate import validate, compare, format_report, metrics

__all__ = ["fit", "random_search", "cma_es", "scipy_local",
           "smc", "SMCResult", "screen", "format_screen",
           "validate", "compare", "format_report", "metrics"]
