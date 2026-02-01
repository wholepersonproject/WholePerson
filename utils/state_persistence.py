"""
State persistence - save and load complete simulation state
"""

import pickle
import json
import numpy as np
from pathlib import Path

def save_state(state, filepath, format='pickle'):
    """
    Save complete simulation state
    
    Args:
        state: SimulationState object
        filepath: Path to save file
        format: 'pickle' (complete) or 'json' (partial)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'pickle':
        # Save complete state (including numpy arrays, agents, etc.)
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"✓ Saved complete state to {filepath}")
    
    elif format == 'json':
        # Save JSON-serializable parts only
        state_dict = {
            'time': state.time,
            'organism': state.organism.copy(),
            'entities': {},
            'organs': {},
            'tissues': {},
        }
        
        # Save entity signals (convert numpy to lists)
        for entity_id, entity in state.entities.items():
            if entity['representation'] == 'lumped':
                state_dict['entities'][entity_id] = {
                    k: float(v) if isinstance(v, np.ndarray) or hasattr(v, 'item') else v
                    for k, v in entity['signals'].items()
                }
            elif entity['representation'] == 'spatial':
                state_dict['entities'][entity_id] = {
                    k: float(v.mean()) if isinstance(v, np.ndarray) else float(v) if hasattr(v, 'item') else v
                    for k, v in entity['signals'].items()
                }
            # Skip agents for JSON (too complex)
        
        # Save organ signals
        for organ_id, organ in state.organs.items():
            if organ.get('representation') == 'lumped':
                state_dict['organs'][organ_id] = {
                    k: float(v) if isinstance(v, np.ndarray) or hasattr(v, 'item') else v
                    for k, v in organ.get('signals', {}).items()
                }
            elif organ.get('representation') == 'spatial':
                state_dict['organs'][organ_id] = {
                    k: float(v.mean()) if isinstance(v, np.ndarray) else float(v) if hasattr(v, 'item') else v
                    for k, v in organ.get('signals', {}).items()
                }
        
        # Save tissue signals
        for tissue_id, tissue in state.tissues.items():
            state_dict['tissues'][tissue_id] = {
                k: float(v) if hasattr(v, 'item') else v
                for k, v in tissue['signals'].items()
            }
        
        # Convert organism state numpy types
        state_dict['organism'] = {
            k: float(v) if hasattr(v, 'item') else v
            for k, v in state.organism.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state_dict, f, indent=2)
        print(f"✓ Saved state summary to {filepath}")


def load_state(filepath):
    """
    Load complete simulation state
    
    Args:
        filepath: Path to saved state file
    
    Returns:
        SimulationState object (if pickle) or dict (if JSON)
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.pkl':
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        print(f"✓ Loaded complete state from {filepath}")
        return state
    
    elif filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            state_dict = json.load(f)
        print(f"✓ Loaded state summary from {filepath}")
        return state_dict
    
    else:
        raise ValueError(f"Unknown format: {filepath.suffix}")


def apply_state_dict(state, state_dict):
    """
    Apply saved state values to a fresh SimulationState
    
    Args:
        state: Fresh SimulationState (from EntityFactory)
        state_dict: Dict from load_state() or save_state(format='json')
    """
    # Set time
    state.time = state_dict.get('time', 0.0)
    
    # Set organism state
    for key, value in state_dict.get('organism', {}).items():
        state.organism[key] = value
    
    # Set entity signals
    for entity_id, signals in state_dict.get('entities', {}).items():
        for signal_name, value in signals.items():
            state.set_signal(entity_id, signal_name, value)
    
    # Set organ signals
    for organ_id, signals in state_dict.get('organs', {}).items():
        for signal_name, value in signals.items():
            state.set_signal(organ_id, signal_name, value)
    
    # Set tissue signals
    for tissue_id, signals in state_dict.get('tissues', {}).items():
        for signal_name, value in signals.items():
            state.set_signal(tissue_id, signal_name, value)
    
    print(f"✓ Applied state values from dict")


# Usage examples:

# 1. Save complete state (pickle - includes everything)
# save_state(state, 'results/final_state.pkl', format='pickle')

# 2. Load and resume simulation
# state = load_state('results/final_state.pkl')
# engine = PhysiologyEngine(state)
# loader = ProcessLoader("configs/processes.yaml")
# loader.load_all_processes(engine)
# engine.run(duration_seconds=24*3600, global_dt=60.0)  # Continue for 24 more hours

# 3. Save as JSON (human-readable, but lossy)
# save_state(state, 'results/final_state.json', format='json')

# 4. Use saved state as initial conditions
# state_dict = load_state('results/final_state.json')
# state = SimulationState()
# factory = EntityFactory("configs/anatomy.yaml")
# factory.initialize_simulation_state(state)
# apply_state_dict(state, state_dict)  # Override with saved values
# engine = PhysiologyEngine(state)
# ...