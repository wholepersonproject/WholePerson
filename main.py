#!/usr/bin/env python3
"""
Baseline simulation without perturbations
"""

from core.state import SimulationState
from core.entity_factory import EntityFactory, ProcessLoader
from engine.engine import PhysiologyEngine
from utils.state_persistence import save_state  # NEW


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


def print_constraint_violations(state):
    """Print any constraint violations that occurred"""
    if not hasattr(state, '_constraints') or not state._constraints:
        return
    
    print("\n" + "="*70)
    print("CONSTRAINT CHECK:")
    print("="*70)
    
    violations = {'critical': [], 'warning': [], 'normal': []}
    
    for (target_id, signal_name), constraint in state._constraints.items():
        value = state.get_signal(target_id, signal_name)
        if value is None:
            value = state.get_organism_state(target_id) if target_id == 'organism' else None
        if value is None:
            continue
        
        # Check critical
        if 'min' in constraint and value <= constraint['min']:
            violations['critical'].append(f"{target_id}.{signal_name} = {value:.1f} (min: {constraint['min']})")
        elif 'max' in constraint and value >= constraint['max']:
            violations['critical'].append(f"{target_id}.{signal_name} = {value:.1f} (max: {constraint['max']})")
        # Check warnings
        elif constraint.get('warn_below') and value < constraint['warn_below']:
            violations['warning'].append(f"{target_id}.{signal_name} = {value:.1f} (warn: {constraint['warn_below']})")
        elif constraint.get('warn_above') and value > constraint['warn_above']:
            violations['warning'].append(f"{target_id}.{signal_name} = {value:.1f} (warn: {constraint['warn_above']})")
        else:
            violations['normal'].append(f"{target_id}.{signal_name} = {value:.1f} ✓")
    
    if violations['critical']:
        print("🔴 CRITICAL:")
        for v in violations['critical']:
            print(f"  {v}")
    
    if violations['warning']:
        print("⚠️  WARNINGS:")
        for v in violations['warning']:
            print(f"  {v}")
    
    print(f"\n✓ Normal: {len(violations['normal'])} signals within bounds")
    print("="*70)


def run_baseline(duration_hours=24):
    print("="*70)
    print("BASELINE SIMULATION")
    print("="*70)
    print()
    
    state = SimulationState()
    state.enforce_constraints = False  # Constraints OFF
    factory = EntityFactory("configs/anatomy.yaml")
    factory.initialize_simulation_state(state)
    print()
    
    engine = PhysiologyEngine(state)
    loader = ProcessLoader("configs/processes.yaml")
    loader.load_all_processes(engine)
    print()
    
    engine.run(duration_seconds=duration_hours * 3600, global_dt=60.0)
    print_full_state(state)
    print_constraint_violations(state)  # NEW: Show constraint violations
    
    return engine, state

if __name__ == "__main__":
    engine, state = run_baseline(500)
    # Save simulation history
    from utils.history import save_simulation
    save_simulation(state, 'results/baseline_sim', formats=['json', 'csv'])
    # Save final state (for resuming)
    save_state(state, 'results/baseline_final.pkl', format='pickle')  # NEW
    save_state(state, 'results/baseline_final.json', format='json')
