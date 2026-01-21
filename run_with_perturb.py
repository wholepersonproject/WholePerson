#!/usr/bin/env python3
"""
Simulation with dietary, exercise, and chronic perturbations
"""

from core.state import SimulationState
from core.entity_factory import EntityFactory, ProcessLoader
from core.perturbation import PerturbationManager
from engine.engine import PhysiologyEngine

def run_with_interventions():
    print("="*70)
    print("SIMULATION WITH PERTURBATIONS")
    print("="*70)
    print()
    
    state = SimulationState()
    factory = EntityFactory("configs/anatomy.yaml")
    factory.initialize_simulation_state(state)
    print()
    
    engine = PhysiologyEngine(state)
    loader = ProcessLoader("configs/processes.yaml")
    loader.load_all_processes(engine)
    print()
    
    perturb_mgr = PerturbationManager("configs/perturbations.yaml")
    engine.set_perturbation_manager(perturb_mgr)
    
    print("Scheduling perturbations:")
    perturb_mgr.add_perturbation("dietary", "mixed_meal", start_time=8*3600)
    perturb_mgr.add_perturbation("exercise", "moderate_aerobic", start_time=10*3600)
    perturb_mgr.add_perturbation("dietary", "mixed_meal", start_time=12*3600)
    perturb_mgr.add_perturbation("dietary", "mixed_meal", start_time=18*3600)
    perturb_mgr.add_perturbation("chronic", "aging", start_time=0)  # Chronic effect
    print()
    
    engine.run(duration_seconds=24 * 3600, global_dt=60.0)
    
    print("="*70)
    print("FINAL STATE:")
    print("="*70)
    print(f"  Blood glucose:  {state.get_signal('blood', 'glucose'):.1f} mg/dL")
    print(f"  Blood insulin:  {state.get_signal('blood', 'insulin'):.1f} µU/mL")
    print(f"  Liver glycogen: {state.get_signal('liver', 'glycogen'):.1f} g")
    print(f"  Beta cell count: {state.entities['beta_cells']['count']}")
    
    # Check if model params changed
    insulin_model = engine.models['insulin_secretion']
    print(f"  Glucose sensitivity: {insulin_model.glucose_sensitivity:.4f} (aging effect)")
    print("="*70)
    print()
    
    return engine, state

if __name__ == "__main__":
    engine, state = run_with_interventions()