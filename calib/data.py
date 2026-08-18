"""
The filesystem edges — the two places calib talks to the outside world.

    load_cgm(path)          CSV on disk           -> (cgm_t, cgm_g)   [IN]
    save_params(best, path) a fitted result       -> YAML overlay     [OUT]

Everything between these two is in-memory. Keeping both here means the rest of
the package never does file I/O, so a different data source (a DB, a DataFrame
you already hold, an API) is a one-function change that touches nothing else.
"""

from __future__ import annotations

import csv
from typing import Dict, Tuple

import numpy as np

# seconds per unit, for the `time_unit` argument of load_cgm
_UNIT_S = {"s": 1.0, "sec": 1.0, "min": 60.0, "m": 60.0, "h": 3600.0, "hr": 3600.0}


def load_cgm(
    path: str,
    time_col: str = "time",
    glucose_col: str = "glucose",
    time_unit: str = "min",
    rebase: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a CGM trace from a CSV into the arrays the Objective expects.

    Returns (cgm_t, cgm_g):
      cgm_t : seconds from the START of the observation window (NOT wall-clock,
              NOT including any warmup — the Objective adds warmup itself).
      cgm_g : glucose in mg/dL.

    The CSV needs a numeric time column and a glucose column, named by
    `time_col`/`glucose_col`. `time_unit` says what the time column is in
    ('s', 'min', 'h'); it's converted to seconds. `rebase=True` shifts the
    first sample to t=0, which is almost always what you want.

    Rows where either field is blank or non-numeric are skipped.

    If your export uses ISO timestamps instead of a numeric offset (Dexcom,
    Libre, etc.), parse them to seconds first and pass time_unit='s' — that
    parsing is device-specific, so it's left to you rather than guessed here.
    """
    if time_unit not in _UNIT_S:
        raise ValueError(f"time_unit must be one of {sorted(_UNIT_S)}, got {time_unit!r}")
    scale = _UNIT_S[time_unit]

    ts, gs = [], []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        for want in (time_col, glucose_col):
            if want not in reader.fieldnames:
                raise ValueError(
                    f"column {want!r} not found; CSV has {reader.fieldnames}"
                )
        for row in reader:
            try:
                t = float(row[time_col]); g = float(row[glucose_col])
            except (TypeError, ValueError):
                continue                      # skip blanks / non-numeric rows
            ts.append(t * scale); gs.append(g)

    if len(ts) < 3:
        raise ValueError(f"{path}: need at least 3 valid rows, got {len(ts)}")

    cgm_t = np.asarray(ts, float)
    cgm_g = np.asarray(gs, float)
    order = np.argsort(cgm_t)                 # tolerate out-of-order rows
    cgm_t, cgm_g = cgm_t[order], cgm_g[order]
    if rebase:
        cgm_t = cgm_t - cgm_t[0]
    return cgm_t, cgm_g


def save_params(best: dict, path: str) -> None:
    """
    Write the fitted parameters back as a YAML overlay that mirrors the
    structure of configs/processes.yaml, so a trained twin is reproducible
    from config. Merge the `processes:` block into your processes.yaml to make
    these the model's new defaults.

        best = fit(obj, cma_es())
        save_params(best, "configs/processes.fitted.yaml")
    """
    import yaml
    if best is None:
        raise ValueError("no successful fit to save")
    overlay: Dict[str, dict] = {}
    for name, value in best["params"].items():
        proc, attr = name.split(".", 1)
        overlay.setdefault(proc, {"parameters": {}})["parameters"][attr] = float(value)
    with open(path, "w") as fh:
        yaml.safe_dump(
            {"processes": overlay,
             "_fit": {"loss": float(best["loss"]),
                      "nuisance": best.get("nuisance", {})}},
            fh, sort_keys=True, default_flow_style=False,
        )
