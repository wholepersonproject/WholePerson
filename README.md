# Whole Person Physiome Models

A multi-scale, mechanistic simulator of whole-body human physiology, built from
the Whole Person Physiome project. It represents the body as a
hierarchy of interacting signals (molecules → tissues → organs → organ systems →
organism) and steps them forward in time with a set of physiological process
models running on their natural timescales (seconds to months).

The current model is strongest on the **endocrine / glucose–insulin axis** and
the **calcium–phosphate–bone axis**, with supporting processes for thyroid,
growth (GH/IGF-1), adrenal steroids, red-cell turnover (EPO), and renal
handling. Roughly 65 processes load by default.

---

## What's in the box

| Path | What it is |
|------|------------|
| `core/state.py` | `SimulationState` — the hierarchical store of all signals, plus a constraint system and history buffer |
| `core/entity_factory.py` | `EntityFactory` builds the body from `anatomy.yaml`; `ProcessLoader` wires up processes from `processes.yaml` |
| `core/graph.py` | `DependencyGraph` — topological ordering of processes each step |
| `core/perturbation.py` | `PerturbationManager` — meals, exercise, drugs, and chronic conditions |
| `engine/engine.py` | `PhysiologyEngine` — multi-timescale stepping, dynamic add/remove of processes, checkpointing |
| `models/` | `ProcessModel` subclasses: `endocrine.py`, `skeletal.py`, `urinary.py` |
| `configs/` | `anatomy.yaml` (organs, tissues, initial values, constraints), `processes.yaml` (which models run, their params & dependencies), `perturbations.yaml` (meal/exercise/drug definitions) |
| `utils/` | `history.py` (`save_simulation` → JSON/CSV), `state_persistence.py` (`save_state`/`load_state`) |
| `calib/` | Modular parameter-fitting scaffold (see `calib/README.md`) |
| `results/` | Pre-computed baseline and three-meals-a-day runs |
| `data/tables/` | Reference physiology spreadsheets, one per organ system |
| `main.py` | Runs a baseline (no perturbations) simulation |
| `run_with_perturb.py` | Runs a 3-meals-a-day scenario, deterministic or randomized |

### How a step works

Each timestep the engine (1) applies any active perturbations, (2) runs every
process in dependency order — but only if enough simulated time has passed for
its timescale — (3) reactivates processes whose inputs now exist, (4) checks
constraints, and (5) advances the clock. A process only reads and writes the
signals it declares in its `inputs`/`outputs`, so the dependency graph stays
explicit and the execution order is computed for you.

---

## Requirements

- Python 3.10+ (tested on 3.12)
- `numpy`, `pyyaml`, `scipy`
- Optional: `cma` — only needed for CMA-ES calibration in `calib/`

```bash
pip install numpy pyyaml scipy
pip install cma          # optional, for calibration
```

Run everything from the repository root (the one containing `core/`, `engine/`,
`models/`, `configs/`) so the config paths resolve.

---

## Quickstart

The fastest way to see it work — a baseline day with no meals:

```bash
python main.py
```

This builds the body, loads the processes, simulates, prints the final blood /
liver / organism state, and writes results to `results/`.

---

## Example: build a body, run it, read a signal

The snippet below is the whole loop in one place. It runs a 6-hour baseline and
reads blood glucose and insulin back out — the values it prints (≈102 mg/dL and
≈13 µU/mL) are the model's fasting steady state.

```python
from core.state import SimulationState
from core.entity_factory import EntityFactory, ProcessLoader
from engine.engine import PhysiologyEngine

# 1. Create the state and (optionally) turn off hard constraint clamping
state = SimulationState()
state.enforce_constraints = False

# 2. Build the anatomy from YAML: organs, tissues, initial signal values
factory = EntityFactory("configs/anatomy.yaml")
factory.initialize_simulation_state(state)

# 3. Create the engine and load the process models
engine = PhysiologyEngine(state)
loader = ProcessLoader("configs/processes.yaml")
loader.load_all_processes(engine)

# 4. Simulate 6 hours at a 60-second timestep, recording every 5 minutes
engine.run(duration_seconds=6 * 3600, global_dt=60.0, record_interval=300)

# 5. Read signals back out
print("blood glucose :", round(state.get_signal("blood", "glucose"), 1), "mg/dL")
print("blood insulin :", round(state.get_signal("blood", "insulin"), 1), "µU/mL")
print("liver glycogen:", round(state.get_signal("liver", "glycogen"), 1))
print("snapshots     :", len(state.history))
```

`get_signal(target, name)` works for blood/molecular entities (`"blood"`),
organs (`"liver"`, `"pancreas"`, `"kidney"`, …), and tissues
(`"muscle_tissue"`, `"adipose_tissue"`). Whole-body state lives in
`state.organism` (e.g. `state.organism["fed_status"]`).

### Adding a meal

Meals (and exercise, drugs, chronic conditions) are perturbations. Attach a
`PerturbationManager`, then fire a meal defined in `configs/perturbations.yaml`.
Here we deliver breakfast three hours in:

```python
from core.perturbation import PerturbationManager

perturb = PerturbationManager("configs/perturbations.yaml")
engine.set_perturbation_manager(perturb)

# ... run a few hours of warm-up first, then:
perturb.add_perturbation(
    "dietary", "breakfast",
    start_time=state.time,
    carb_grams=45,     # optional override of the YAML default
    peak_time=12.0,    # optional: minutes to gastric peak (faster eating = higher spike)
)
```

For a full day of breakfast/lunch/dinner — including a randomized version where
carbs, eating speed, and timing vary day to day — see `run_with_perturb.py`:

```bash
python run_with_perturb.py
```

### Saving and resuming

```python
from utils.history import save_simulation
from utils.state_persistence import save_state, load_state

# Full time series -> results/mysim.json and results/mysim.csv
save_simulation(state, "results/mysim", formats=["json", "csv"])

# Final state only, for resuming later
save_state(state, "results/mysim_final.pkl", format="pickle")
state = load_state("results/mysim_final.pkl")
```

---

## Configuration, not code

Most changes don't require touching Python:

- **Change the body** — edit `configs/anatomy.yaml` to add an organ, change an
  initial value, or set a constraint (`min` / `max` clamp, `warn_below` /
  `warn_above` alert).
- **Change what runs** — edit `configs/processes.yaml` to add/remove a process,
  retune its parameters, or change its dependencies and timescale
  (`seconds` / `minutes` / `hours` / `days` / `months`).
- **Change the inputs** — edit `configs/perturbations.yaml` to define new meals,
  workouts, or drug doses.

### Adding a new process in code

Subclass `ProcessModel`, declare the signals you read and write, and implement
`step`:

```python
from models.base import ProcessModel, TimeScale

class MyProcess(ProcessModel):
    inputs  = {"glucose": ("blood", "glucose")}
    outputs = {"marker":  ("blood", "my_marker")}

    def __init__(self, rate=0.1):
        super().__init__("my_process", TimeScale.MINUTES)
        self.rate = rate

    def step(self, state, dt):
        g = state.get_signal("blood", "glucose")
        state.set_signal("blood", "my_marker", g * self.rate)

engine.register_model("my_process", MyProcess(), dependencies=[])
```

The engine auto-creates missing output signals and recomputes execution order
when you register or remove a process — even mid-run.

---

## Calibrating to data

`calib/` fits model parameters to a known meal log + CGM trace. The honest
workflow is *screen → priors → SMC* (identify what your data can constrain,
then get a posterior), but you can start with a point estimate:

```bash
python calib/examples/point_estimate.py
```

That example generates a synthetic CGM from known parameters and estimates them
back, so you can watch the whole loop on one screen. See `calib/README.md` for
the full story, including bringing your own Dexcom/Libre data.

---

## Notes

- Each `model.simulate` in `calib/` rebuilds the state and engine from scratch —
  several processes carry hidden state across steps, so reusing an engine would
  make results depend on evaluation order. Budget ~7 s per simulated day on one
  core.
- `results/` ships with a completed `baseline_sim` and `three_meals_sim` you can
  inspect without running anything.
- `data/tables/data/` holds the draft reference spreadsheets the process models
  were parameterized against, one per organ system.

