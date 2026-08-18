"""
Sensitivity screen — run this BEFORE choosing what to estimate.

Answers two questions with a handful of simulations:

  1. Which parameters actually move the CGM?  (rank by response magnitude)
     Parameters the trace barely responds to are not identifiable from CGM;
     don't put them in the free set.

  2. Which parameters move it the SAME WAY?  (pairwise response correlation)
     Two parameters whose CGM-response vectors are nearly parallel are
     degenerate — only their combination is identifiable, never both alone.
     This is the muscle-vs-adipose problem; ranking singles alone misses it,
     which is the whole reason this step exists.

Cost: 2 sims per candidate parameter (a + and a - perturbation) plus one
baseline. Cheap. Do it first, every time.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..objective import Objective


def screen(obj: Objective, step: float = 0.15) -> dict:
    """
    One-at-a-time perturbation of each free parameter around the defaults.

    `step` is in unit-cube coordinates (fraction of each parameter's range).
    Returns a dict with per-parameter sensitivity and the pairwise response
    correlation matrix. Print it with `print(format_screen(result))`.
    """
    x0 = obj.x0()
    d = len(obj.free)
    names = [p.name for p in obj.free]

    # baseline predicted CGM
    base_resid, _, _ = obj.residuals(x0)
    if base_resid is None:
        raise RuntimeError("baseline simulation failed; fix the model before screening")
    base_pred = base_resid + obj.cgm_g          # recover pred = resid + obs

    responses = np.zeros((d, obj.cgm_t.size))   # CGM-response vector per param
    sensitivity = np.zeros(d)

    for i in range(d):
        xp, xm = x0.copy(), x0.copy()
        xp[i] = min(1.0, x0[i] + step)
        xm[i] = max(0.0, x0[i] - step)

        rp, _, _ = obj.residuals(xp)
        rm, _, _ = obj.residuals(xm)
        if rp is None or rm is None:
            responses[i] = np.nan
            sensitivity[i] = np.nan
            continue

        pred_p = rp + obj.cgm_g
        pred_m = rm + obj.cgm_g
        resp = pred_p - pred_m                   # how the trace shifts, +vs-
        responses[i] = resp
        sensitivity[i] = float(np.sqrt(np.mean(resp ** 2)))   # RMS trace shift

    # pairwise correlation of response vectors -> degeneracy detector
    corr = np.full((d, d), np.nan)
    for i in range(d):
        for j in range(d):
            a, b = responses[i], responses[j]
            if np.isfinite(a).all() and np.isfinite(b).all() and a.std() > 0 and b.std() > 0:
                corr[i, j] = float(np.corrcoef(a, b)[0, 1])

    # flag degenerate pairs
    ref = np.nanmax(sensitivity) if np.isfinite(sensitivity).any() else 1.0
    degenerate: List[Tuple[str, str, float]] = []
    for i in range(d):
        for j in range(i + 1, d):
            if np.isfinite(corr[i, j]) and abs(corr[i, j]) > 0.98:
                degenerate.append((names[i], names[j], corr[i, j]))

    return {
        "names": names,
        "sensitivity": sensitivity,
        "corr": corr,
        "degenerate_pairs": degenerate,
        "ref": ref,
        "step": step,
    }


def format_screen(r: dict) -> str:
    names, sens, ref = r["names"], r["sensitivity"], r["ref"]
    order = np.argsort(-np.nan_to_num(sens))
    lines = ["SENSITIVITY (RMS shift in predicted CGM, mg/dL, for +/- %.2f in unit space)"
             % r["step"], "-" * 68]
    for k in order:
        s = sens[k]
        if not np.isfinite(s):
            verdict = "UNSTABLE (sim failed)"
        elif s < 0.02 * ref:
            verdict = "flat — NOT identifiable, leave fixed"
        elif s < 0.15 * ref:
            verdict = "weak"
        else:
            verdict = "informative — keep free"
        lines.append(f"  {names[k]:<48} {s:8.2f}   {verdict}")

    lines.append("")
    if r["degenerate_pairs"]:
        lines.append("DEGENERATE PAIRS (|response correlation| > 0.98):")
        for a, b, c in r["degenerate_pairs"]:
            lines.append(f"  {a}")
            lines.append(f"  {b}   corr={c:+.3f}")
            lines.append("    -> identifiable only in combination; free ONE, fix/lump the other")
    else:
        lines.append("No degenerate pairs among candidates (all responses distinct).")

    lines.append("")
    lines.append("Rule: free the 'informative' params, fix the 'flat' ones, and from each")
    lines.append("degenerate pair keep only one. That set is what a single CGM can identify.")
    return "\n".join(lines)
