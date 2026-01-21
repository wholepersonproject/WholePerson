#!/usr/bin/env python3
"""
Baseline simulation without perturbations
"""

from core.state import SimulationState
from core.entity_factory import EntityFactory, ProcessLoader
from engine.engine import PhysiologyEngine

def print_full_state(state):
    """Print all important signals"""
    print("="*70)
    print("STATE:")
    print("="*70)
    
    # Blood
    print("BLOOD:")
    for signal in ['glucose', 'insulin', 'glucagon', 'pancreatic_polypeptide']:
        val = state.get_signal('blood', signal)
        if val: print(f"  {signal}: {val:.1f}")
    
    # Liver
    print("LIVER:")
    for signal in ['glycogen', 'glucose']:
        val = state.get_signal('liver', signal)
        if val: print(f"  {signal}: {val:.1f}")
    
    # Organism
    print("ORGANISM:")
    for key, val in state.organism.items():
        print(f"  {key}: {val}")
    
    print("="*70)

def run_baseline(duration_hours=24):
    print("="*70)
    print("BASELINE SIMULATION")
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
    
    engine.run(duration_seconds=duration_hours * 3600, global_dt=60.0)
    print_full_state(state)
    
    return engine, state

if __name__ == "__main__":
    engine, state = run_baseline(24)
    # Save simulation history
    from utils.history import save_simulation
    save_simulation(state, 'results/baseline_sim', formats=['json', 'csv'])