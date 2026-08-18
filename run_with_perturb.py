#!/usr/bin/env python3
"""
Simulation with 3 meals/day (breakfast, lunch, dinner).

Meals can be deterministic (fixed carbs at fixed times) or randomized.
Randomization draws, per meal per day:
  - carbohydrate load  ~ Normal(mean, sd), clipped to a floor   -> sets gastric spike peak_magnitude
  - eating speed        ~ Uniform(min, max) minutes-to-peak     -> sets gastric spike peak_time
  - meal timing         ~ Normal(0, sd_h) jitter, clipped        -> shifts the fire time
Both carbs and peak_time are pushed into the meal via PerturbationManager.add_perturbation.
"""
import numpy as np

from core.state import SimulationState
from core.entity_factory import EntityFactory, ProcessLoader
from core.perturbation import PerturbationManager
from engine.engine import PhysiologyEngine
from utils.history import save_simulation

MEAL_SCHEDULE_HOURS = [7, 12, 18]          # breakfast, lunch, dinner (nominal)
MEAL_NAMES = ['breakfast', 'lunch', 'dinner']

# per-meal nominal carbohydrate load (g) and its day-to-day spread (g)
MEAL_CARB_MEAN = {'breakfast': 40.0, 'lunch': 60.0, 'dinner': 70.0}
MEAL_CARB_SD   = {'breakfast': 8.0,  'lunch': 12.0, 'dinner': 14.0}

# eating speed: minutes for the stomach to reach peak fill (lower = faster eating = higher peak)
PEAK_TIME_RANGE = (8.0, 25.0)
# timing jitter on when the meal is actually eaten (hours), clipped to +/- 1 h
TIMING_JITTER_SD_H = 0.5
CARB_FLOOR_G = 15.0


def run_three_meals_a_day(duration_hours=500, global_dt=60.0,
                          randomize_meals=True, meal_seed=1):
    """Run the 3-meals-a-day scenario.

    randomize_meals=False -> deterministic fixed meals (original behaviour).
    randomize_meals=True  -> per-meal random carbs, eating speed, and timing.
    meal_seed             -> makes a randomized run reproducible; change it for a new week.
    """
    rng = np.random.default_rng(meal_seed)

    state = SimulationState()
    factory = EntityFactory("configs/anatomy.yaml")
    factory.initialize_simulation_state(state)

    engine = PhysiologyEngine(state)
    loader = ProcessLoader("configs/processes.yaml")
    loader.load_all_processes(engine)

    perturb_mgr = PerturbationManager("configs/perturbations.yaml")
    engine.set_perturbation_manager(perturb_mgr)

    # Pre-draw a meal plan so each (day, meal) has a fixed, reproducible draw.
    ndays = int(duration_hours // 24) + 1
    plan = {}  # (day, meal_name) -> dict(fire_hour, carbs, peak_time)
    for day in range(ndays):
        for meal_hour, meal_name in zip(MEAL_SCHEDULE_HOURS, MEAL_NAMES):
            if randomize_meals:
                jitter = float(np.clip(rng.normal(0.0, TIMING_JITTER_SD_H), -1.0, 1.0))
                carbs = float(np.clip(rng.normal(MEAL_CARB_MEAN[meal_name],
                                                 MEAL_CARB_SD[meal_name]), CARB_FLOOR_G, None))
                ptime = float(rng.uniform(*PEAK_TIME_RANGE))
                plan[(day, meal_name)] = {'fire_hour': meal_hour + jitter,
                                          'carbs': carbs, 'peak_time': ptime}
            else:
                plan[(day, meal_name)] = {'fire_hour': float(meal_hour),
                                          'carbs': None, 'peak_time': None}

    target_time = duration_hours * 3600
    scheduled = set()  # (day, meal_name) already fired

    while state.time < target_time:
        day = int(state.time // 86400)
        hour_of_day = (state.time % 86400) / 3600.0

        for meal_name in MEAL_NAMES:
            key = (day, meal_name)
            if key in scheduled or key not in plan:
                continue
            entry = plan[key]
            if hour_of_day >= entry['fire_hour']:
                perturb_mgr.add_perturbation(
                    "dietary", meal_name, start_time=state.time,
                    carb_grams=entry['carbs'], peak_time=entry['peak_time'])
                scheduled.add(key)

        engine.step(global_dt)

        if state.time % 30 < global_dt:  # record every 5 min, matches engine.run's default
            state.record_history()

    state.record_history()
    return engine, state


if __name__ == "__main__":
    # Deterministic (original): randomize_meals=False
    # Random meals: set randomize_meals=True (change meal_seed for a different week)
    engine, state = run_three_meals_a_day(164, randomize_meals=True, meal_seed=1)
    save_simulation(state, 'results/three_meals_sim', formats=['json', 'csv'])