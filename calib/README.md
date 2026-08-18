# calib — modular parameter fitting for PhysiomeTwin

Known meals + known CGM → estimate parameters. The layout separates the part
that never changes (the objective contract) from the part you're still
choosing (the method), so you can start fitting before you've committed to how.

## Layout

```
calib/
  model.py        ┐  THE CORE — the stable f(x) contract. Read these four
  observe.py      │  files and you understand the whole thing.
  objective.py    │  objective.py is Objective + Param, nothing else.
  priors.py       ┘
  methods/        ← THE ANNEX — one file per method; add/pick without
    optimize.py     touching the core.
    sample.py         optimize: fit, random_search, cma_es, scipy_local
    screen.py         sample:   smc, SMCResult
    validate.py       screen:   screen, format_screen        (run FIRST)
                      validate: validate, compare, format_report (run LAST)
  data.py         ← THE EDGES — the only disk I/O: load_cgm, save_params
  examples/       ← runnable end-to-end scripts
    point_estimate.py   bayesian_smc.py   screen_all.py
```

Drop `calib/` into the PhysiomeTwin repo root (next to `core/`, `engine/`,
`models/`, `configs/`), then:

```bash
python calib/examples/point_estimate.py
```

The example makes a synthetic CGM from known parameters and estimates them
back, so you can watch the whole loop run on one screen.

## The seams

```
model.simulate       params + protocol -> glucose trace       (SEAM 1)
observe.Observation  plasma glucose    -> predicted CGM        (SEAM 2)
objective.Objective  x in [0,1]^d      -> scalar / logL / logP (SEAM 3)
methods/*            anything driving f(x)                     (SEAM 4)
```

The Objective exposes a loss and a Bayesian interface off the SAME
simulate-and-align core, so one seam feeds every method family:

    obj(x)                 -> loss   (RMSE/MAE)  -> point optimizers, ABC distance
    obj.log_likelihood(x)  -> Gaussian logL      -> MLE view
    obj.log_prior(x)       -> joint log prior    -> from priors.PriorSet
    obj.log_posterior(x)   -> logL + log_prior   -> MCMC / SMC samplers

| Swap this…                | Seam | …for |
|---------------------------|------|------|
| `model.py`                | 1    | a different simulator — the only file that imports PhysiomeTwin |
| `observe.py`              | 2    | pure interpolation, a two-compartment sensor, a learned model |
| `objective.py`            | 3    | a different loss, a regulariser, multi-signal residuals |
| a file in `methods/`      | 4    | scipy, CMA-ES, Optuna, a sampler — anything calling `f(x)` |

## Haven't picked a method yet?

That's the point of the split. Write your objective against the core, and the
choice of method stays a one-import decision:

```python
from calib import Objective, fit, cma_es          # point estimate
best = fit(obj, cma_es(n_evals=150))

from calib import smc, HalfNormal                  # or a posterior
res = smc(obj, sigma_prior=HalfNormal(sd=15))
```

To add a method that doesn't exist yet, drop one file in `methods/` that
consumes `objective.Objective`. Nothing in the core changes.

## Recommended workflow for one observable: screen → priors → SMC

With a single CGM you can identify only a handful of parameters. The honest
pipeline (see `examples/bayesian_smc.py`):

1. **`screen(obj, step)`** — perturbs each candidate, ranks by how much the CGM
   moves, and reports the pairwise response-correlation matrix. Free the
   `informative` parameters, leave `flat` ones fixed, and from any degenerate
   pair (|corr| > 0.98) keep only one. `print(format_screen(screen(obj)))`.
2. **Pick the free set** from the screen — the tool advises, you decide.
3. **`PriorSet`** — literature priors: `Uniform(lo, hi)` for a range only,
   `Normal(mu, sd)` for a central value with a spread.
4. **`smc(obj, sigma_prior=HalfNormal(sd=15))`** — returns a posterior (not a
   point), with the CGM noise SD `sigma` estimated jointly. `res.summary()`
   flags each marginal as `data-driven` (identified) or `prior-driven` (the
   CGM said little).

`examples/screen_all.py` screens the whole glucose pathway (~72 parameters) in
one run to turn "there are a lot of params" into a data-backed free set.

## Priors (priors.py)

A prior on one parameter is just a `log_prob(x)`; the joint prior is the sum.

```python
from calib import PriorSet, Normal, Uniform, LogNormal
prior = PriorSet({
    "insulin_secretion.glucose_sensitivity":     Normal(1.0, 0.3),
    "glucose_uptake_muscle.insulin_sensitivity": Uniform(0.2, 2.5),
    "hepatic_glucose_production.production_rate": LogNormal.from_median_cv(2.0, 0.25),
})
obj = Objective(..., sigma=8.0, prior=prior)   # sigma = CGM noise sd, mg/dL
```

Available: `Uniform`, `Normal`, `LogNormal`, `TruncatedNormal`, `HalfNormal`.
Add your own by copying the pattern — return the log-density, `-inf` off-support.
`sigma` is a positive scale, so its prior is `HalfNormal` (or `LogNormal`) —
never `Normal`.

## Bringing your own data (data.py)

```python
from calib import load_cgm
cgm_t, cgm_g = load_cgm("mydata.csv", time_col="t_min", glucose_col="mg_dl",
                        time_unit="min")   # cgm_t comes back in seconds-from-start
```

`cgm_t` is seconds from the start of the observation window (not wall-clock,
not including warmup — the Objective adds warmup itself). ISO-timestamp exports
(Dexcom/Libre) are device-specific; parse them to seconds yourself and pass
`time_unit="s"`.

Then set `protocol` to your real meal log — a list of
`(t_seconds, carb_grams, peak_time_min)` — and list the parameters to fit as
`Param("process_id.attr", lo, hi)`.

## Saving results

```python
from calib import save_params
save_params(best, "configs/processes.fitted.yaml")
```

Writes a YAML overlay mirroring `processes.yaml`. Merge its `processes:` block
into `processes.yaml` to make the fitted values the model's new defaults — the
one write back into the repo; everything else runs in memory.

## Validating a fit (methods/validate.py)

Fitting RMSE always looks good — the optimizer minimised it. The honest check
is baseline-vs-fitted, ideally on a held-out window:

```python
from calib import compare, format_report
print(format_report(compare(obj, best["params"])))   # defaults vs fitted
```

```
GLUCOSE VALIDATION  (n=120 CGM samples)
------------------------------------------------------------
metric                   baseline       fitted
  RMSE (mg/dL)               7.30         6.40
  MAE  (mg/dL)               5.81         5.02
  bias (mg/dL)              +1.90        +0.12
  correlation               0.912        0.964
------------------------------------------------------------
RMSE improvement: +12.3%  (baseline -> fitted, better)
```

For a real test of generalisation, build a SECOND Objective on a held-out
CGM+protocol (same `free` list) and pass the fitted params to it — see
`examples/validate.py`. `validate` reads the same `Objective.residuals` core the
loss does, so these numbers can't disagree with what the optimizer minimised.

## Notes / knobs

- **Fresh build every call.** `model.simulate` rebuilds state + engine each
  time, because several processes carry hidden state across steps; reusing an
  engine would make the loss depend on evaluation order.
- **Cost.** ~7 s per simulated day, single core. The two cost knobs are
  `duration_s` and the driver's evaluation budget.
- **`random_search` won't fit anything** — it exists only to exercise the seam.
  Use `cma_es` (needs `pip install cma`, and >= 2 free params) for a real fit.
- **SMC tuning.** `n_particles >= 200` for a real posterior; `move_steps` 2–5
  (never 0 — the move kernel is what restores particle diversity after
  resampling; without it the population collapses and reports a fake
  zero-width posterior); `move_scale ~0.1`. This mini SMC is serial — evaluate
  the within-stage particles across a process pool to scale.
- Deliberately omitted: caching, parallelism, data QC, and identifiability
  analysis beyond the screen. Add them at the seam that needs them.
```
