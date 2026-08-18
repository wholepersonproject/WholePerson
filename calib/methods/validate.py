"""
Post-fit validation — how good is a parameter set, in interpretable units.

Reads the SAME simulate-and-align core as the loss (`Objective.residuals`), so
the numbers here can NEVER disagree with what the optimizer minimised. This is
the one guarantee a separate validator (like the repo's old training/ one, which
re-ran the engine its own way) can't give you.

    validate(obj, params)          one parameter set -> {rmse, mae, bias, corr}
    compare(obj, fitted, baseline) baseline vs fitted + improvement
    format_report(cmp)             printable before/after table

Held-out validation is the honest test: fitting RMSE always looks good because
the optimizer minimised it. Build a SECOND Objective on a held-out CGM+protocol
(same `free` list) and pass your fitted params to it — see examples/validate.py.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..objective import Objective


def _params_to_x(obj: Objective, params: Dict[str, float]) -> np.ndarray:
    """Map a {name: natural-value} dict to obj's unit-cube x. Any free
    parameter missing from `params` falls back to the objective's default."""
    x = np.empty(len(obj.free))
    for i, p in enumerate(obj.free):
        v = params.get(p.name, obj.defaults.get(p.name))
        if v is None:
            raise KeyError(f"no value for {p.name!r} in params or defaults")
        x[i] = p.to_unit(v)
    return np.clip(x, 0.0, 1.0)


def metrics(resid: np.ndarray, obs: np.ndarray) -> Dict[str, float]:
    """Interpretable error metrics from residuals (pred - obs) and the CGM.
    bias = mean(pred - obs): positive means the model runs high."""
    resid = np.asarray(resid, float)
    obs = np.asarray(obs, float)
    pred = resid + obs
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    bias = float(np.mean(resid))
    if pred.std() > 0 and obs.std() > 0:
        corr = float(np.corrcoef(pred, obs)[0, 1])
    else:
        corr = float("nan")               # constant trace -> correlation undefined
    return {"rmse": rmse, "mae": mae, "bias": bias, "corr": corr, "n": int(resid.size)}


def validate(obj: Objective, params: Optional[Dict[str, float]] = None) -> dict:
    """
    Evaluate ONE parameter set against obj's CGM. `params=None` uses the
    objective's defaults (the baseline). Returns {ok, params, metrics, nuisance}.
    A failed simulation returns ok=False and metrics=None rather than raising,
    so a broken baseline still reports cleanly.
    """
    x = obj.x0() if params is None else _params_to_x(obj, params)
    resid, ov, nuis = obj.residuals(x)
    if resid is None:
        return {"ok": False, "params": ov, "metrics": None, "nuisance": None}
    return {"ok": True, "params": ov, "metrics": metrics(resid, obj.cgm_g),
            "nuisance": nuis}


def compare(
    obj: Objective,
    fitted_params: Dict[str, float],
    baseline_params: Optional[Dict[str, float]] = None,
) -> dict:
    """
    Baseline vs fitted on the SAME objective. `baseline_params=None` uses the
    defaults. Pass an objective built on a HELD-OUT window to test whether the
    fit generalises. Returns both metric sets plus the RMSE improvement %.
    """
    base = validate(obj, baseline_params)
    post = validate(obj, fitted_params)
    improvement = None
    if base["ok"] and post["ok"]:
        b = base["metrics"]["rmse"]
        p = post["metrics"]["rmse"]
        improvement = None if b == 0 else (b - p) / b * 100.0
    return {
        "baseline": base,
        "post": post,
        "rmse_improvement_pct": improvement,
        "n": post["metrics"]["n"] if post["ok"] else None,
    }


def format_report(cmp: dict) -> str:
    """Pretty-print a `compare()` result as a baseline-vs-fitted table."""
    base, post = cmp["baseline"], cmp["post"]
    n = cmp.get("n")
    head = f"GLUCOSE VALIDATION" + (f"  (n={n} CGM samples)" if n else "")
    lines = [head, "-" * 60,
             f"{'metric':<20}{'baseline':>13}{'fitted':>13}"]

    def cell(res, key, fmt):
        if not res["ok"]:
            return f"{'sim failed':>13}"
        v = res["metrics"][key]
        return f"{format(v, fmt):>13}"

    for key, label, fmt in [("rmse", "RMSE (mg/dL)", "8.2f"),
                            ("mae", "MAE  (mg/dL)", "8.2f"),
                            ("bias", "bias (mg/dL)", "+8.2f"),
                            ("corr", "correlation", "8.3f")]:
        lines.append(f"  {label:<18}{cell(base, key, fmt)}{cell(post, key, fmt)}")

    lines.append("-" * 60)
    imp = cmp["rmse_improvement_pct"]
    if imp is None:
        lines.append("RMSE improvement: n/a (a simulation failed)")
    else:
        arrow = "better" if imp >= 0 else "WORSE"
        lines.append(f"RMSE improvement: {imp:+.1f}%  (baseline -> fitted, {arrow})")
    return "\n".join(lines)
