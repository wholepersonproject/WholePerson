"""
Utilities for saving and loading simulation history
"""
import json
import pickle
import numpy as np
from pathlib import Path

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class HistoryLogger:
    """Save simulation history to disk"""
    
    @staticmethod
    def save_json(state, filepath):
        """
        Save history as JSON (human-readable)
        
        Args:
            state: SimulationState with history
            filepath: Where to save (e.g., "results/sim_001.json")
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON
        history_serializable = []
        for snapshot in state.history:
            snap_copy = {
                'time': float(snapshot['time']),
                'entities': {},
                'tissues': {},
                'organs': {},
                'organ_systems': {},
                'organism': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                           for k, v in snapshot['organism'].items()}
            }
            
            # Entities
            for entity_id, signals in snapshot['entities'].items():
                snap_copy['entities'][entity_id] = {}
                for signal_name, value in signals.items():
                    if isinstance(value, np.ndarray):
                        snap_copy['entities'][entity_id][signal_name] = value.tolist()
                    elif isinstance(value, (np.floating, np.integer)):
                        snap_copy['entities'][entity_id][signal_name] = float(value)
                    else:
                        snap_copy['entities'][entity_id][signal_name] = value
            
            # Tissues
            for tissue_id, signals in snapshot.get('tissues', {}).items():
                snap_copy['tissues'][tissue_id] = {}
                for signal_name, value in signals.items():
                    if isinstance(value, (np.floating, np.integer)):
                        snap_copy['tissues'][tissue_id][signal_name] = float(value)
                    else:
                        snap_copy['tissues'][tissue_id][signal_name] = value
            
            # Organs
            for organ_id, signals in snapshot.get('organs', {}).items():
                snap_copy['organs'][organ_id] = {}
                for signal_name, value in signals.items():
                    if isinstance(value, (np.floating, np.integer)):
                        snap_copy['organs'][organ_id][signal_name] = float(value)
                    else:
                        snap_copy['organs'][organ_id][signal_name] = value
            
            # Organ systems
            for system_name, states in snapshot.get('organ_systems', {}).items():
                snap_copy['organ_systems'][system_name] = {}
                for state_name, value in states.items():
                    if isinstance(value, (np.floating, np.integer)):
                        snap_copy['organ_systems'][system_name][state_name] = float(value)
                    else:
                        snap_copy['organ_systems'][system_name][state_name] = value
            
            history_serializable.append(snap_copy)
        
        with open(filepath, 'w') as f:
            json.dump(history_serializable, f, indent=2, cls=NumpyEncoder)
        
        print(f"✓ Saved {len(state.history)} snapshots to {filepath}")
    
    @staticmethod
    def save_pickle(state, filepath):
        """
        Save history as pickle (preserves numpy arrays, faster)
        
        Args:
            state: SimulationState with history
            filepath: Where to save (e.g., "results/sim_001.pkl")
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(state.history, f)
        
        print(f"✓ Saved {len(state.history)} snapshots to {filepath}")
    
    @staticmethod
    def save_csv(state, filepath, signals=None):
        """
        Save history as CSV (for Excel/analysis)
        
        Args:
            state: SimulationState with history
            filepath: Where to save (e.g., "results/sim_001.csv")
            signals: List of (type, id, signal) tuples to save
                    type can be 'entity', 'organ', 'tissue', or 'organism'
                    If None, automatically saves ALL signals from state
        
        Example:
            signals = [
                ('entity', 'blood', 'glucose'),
                ('entity', 'blood', 'insulin'),
                ('organ', 'liver', 'glycogen'),
                ('organ', 'kidney', 'epo_production_capacity'),
                ('organism', None, 'age')
            ]
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect all signals if not specified
        if signals is None:
            signals = []
            
            # Get first snapshot to detect structure
            if len(state.history) > 0:
                snapshot = state.history[0]
                
                # All entity signals
                for entity_id, entity_signals in snapshot.get('entities', {}).items():
                    for signal_name in entity_signals.keys():
                        signals.append(('entity', entity_id, signal_name))
                
                # All organ signals
                for organ_id, organ_signals in snapshot.get('organs', {}).items():
                    for signal_name in organ_signals.keys():
                        signals.append(('organ', organ_id, signal_name))
                
                # All tissue signals
                for tissue_id, tissue_signals in snapshot.get('tissues', {}).items():
                    for signal_name in tissue_signals.keys():
                        signals.append(('tissue', tissue_id, signal_name))
                
                # All organ system signals
                for system_id, system_signals in snapshot.get('organ_systems', {}).items():
                    for signal_name in system_signals.keys():
                        signals.append(('organ_system', system_id, signal_name))
                
                # All organism state
                for state_name in snapshot.get('organism', {}).keys():
                    signals.append(('organism', None, state_name))
            
            print(f"  Auto-detected {len(signals)} signals to save")
        
        # Build CSV manually (no pandas dependency)
        lines = []
        
        # Header
        header = ['time_seconds', 'time_hours']
        for item in signals:
            if len(item) == 2:
                # Old format: (entity_id, signal_name) - assume entity
                entity_id, signal_name = item
                header.append(f"{entity_id}.{signal_name}")
            elif len(item) == 3:
                # New format: (type, id, signal_name)
                item_type, item_id, signal_name = item
                if item_type == 'organism':
                    header.append(f"organism.{signal_name}")
                elif item_type == 'organ_system':
                    header.append(f"organ_system.{item_id}.{signal_name}")
                else:
                    header.append(f"{item_type}.{item_id}.{signal_name}")
        lines.append(','.join(header))
        
        # Data rows
        for snapshot in state.history:
            row = [
                str(snapshot['time']),
                str(snapshot['time'] / 3600)
            ]
            
            for item in signals:
                if len(item) == 2:
                    # Old format: (entity_id, signal_name)
                    entity_id, signal_name = item
                    if entity_id in snapshot['entities']:
                        if signal_name in snapshot['entities'][entity_id]:
                            value = snapshot['entities'][entity_id][signal_name]
                            try:
                                row.append(str(float(value)))
                            except (ValueError, TypeError):
                                row.append(str(value))
                        else:
                            row.append('')
                    else:
                        row.append('')
                
                elif len(item) == 3:
                    # New format: (type, id, signal_name)
                    item_type, item_id, signal_name = item
                    
                    if item_type == 'entity':
                        if item_id in snapshot.get('entities', {}):
                            value = snapshot['entities'][item_id].get(signal_name, '')
                            if value != '':
                                try:
                                    row.append(str(float(value)))
                                except (ValueError, TypeError):
                                    row.append(str(value))  # Keep strings as-is
                            else:
                                row.append('')
                        else:
                            row.append('')
                    
                    elif item_type == 'organ':
                        if item_id in snapshot.get('organs', {}):
                            value = snapshot['organs'][item_id].get(signal_name, '')
                            if value != '':
                                try:
                                    row.append(str(float(value)))
                                except (ValueError, TypeError):
                                    row.append(str(value))
                            else:
                                row.append('')
                        else:
                            row.append('')
                    
                    elif item_type == 'tissue':
                        if item_id in snapshot.get('tissues', {}):
                            value = snapshot['tissues'][item_id].get(signal_name, '')
                            if value != '':
                                try:
                                    row.append(str(float(value)))
                                except (ValueError, TypeError):
                                    row.append(str(value))
                            else:
                                row.append('')
                        else:
                            row.append('')
                    
                    elif item_type == 'organ_system':
                        if item_id in snapshot.get('organ_systems', {}):
                            value = snapshot['organ_systems'][item_id].get(signal_name, '')
                            if value != '':
                                try:
                                    row.append(str(float(value)))
                                except (ValueError, TypeError):
                                    row.append(str(value))
                            else:
                                row.append('')
                        else:
                            row.append('')
                    
                    elif item_type == 'organism':
                        value = snapshot.get('organism', {}).get(signal_name, '')
                        if value != '':
                            try:
                                row.append(str(float(value)))
                            except (ValueError, TypeError):
                                row.append(str(value))  # Keep strings like 'fasted'
                        else:
                            row.append('')
            
            lines.append(','.join(row))
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"✓ Saved {len(state.history)} snapshots to {filepath}")
        print(f"  Columns: {header}")
    
    @staticmethod
    def load_json(filepath):
        """Load history from JSON"""
        with open(filepath, 'r') as f:
            history = json.load(f)
        print(f"✓ Loaded {len(history)} snapshots from {filepath}")
        return history
    
    @staticmethod
    def load_pickle(filepath):
        """Load history from pickle"""
        with open(filepath, 'rb') as f:
            history = pickle.load(f)
        print(f"✓ Loaded {len(history)} snapshots from {filepath}")
        return history


class HistoryAnalyzer:
    """Analyze simulation history"""
    
    @staticmethod
    def to_dataframe(state, signals=None):
        """
        Convert history to pandas DataFrame (requires pandas)
        
        Args:
            state: SimulationState with history
            signals: List of (type, id, signal) tuples
                    If None, uses common entity signals
        
        Example:
            signals = [
                ('entity', 'blood', 'glucose'),
                ('organ', 'kidney', 'epo_production_capacity'),
                ('organism', None, 'age')
            ]
        """
        try:
            import pandas as pd
        except ImportError:
            print("ERROR: pandas not installed. Install with: pip install pandas")
            return None
        
        if signals is None:
            signals = [
                ('entity', 'blood', 'glucose'),
                ('entity', 'blood', 'insulin'),
                ('entity', 'blood', 'glucagon'),
                ('entity', 'liver', 'glycogen')
            ]
        
        data = {'time': []}
        
        # Initialize columns based on signal format
        for item in signals:
            if len(item) == 2:
                # Old format: (entity_id, signal_name)
                entity_id, signal_name = item
                data[f"{entity_id}_{signal_name}"] = []
            elif len(item) == 3:
                # New format: (type, id, signal_name)
                item_type, item_id, signal_name = item
                if item_type == 'organism':
                    data[f"organism_{signal_name}"] = []
                elif item_type == 'organ_system':
                    data[f"organ_system_{item_id}_{signal_name}"] = []
                else:
                    data[f"{item_type}_{item_id}_{signal_name}"] = []
        
        for snapshot in state.history:
            data['time'].append(snapshot['time'] / 3600)  # hours
            
            for item in signals:
                if len(item) == 2:
                    # Old format
                    entity_id, signal_name = item
                    col_name = f"{entity_id}_{signal_name}"
                    if entity_id in snapshot.get('entities', {}):
                        value = snapshot['entities'][entity_id].get(signal_name, np.nan)
                        data[col_name].append(value)
                    else:
                        data[col_name].append(np.nan)
                
                elif len(item) == 3:
                    # New format
                    item_type, item_id, signal_name = item
                    
                    if item_type == 'organism':
                        col_name = f"organism_{signal_name}"
                        value = snapshot.get('organism', {}).get(signal_name, np.nan)
                        data[col_name].append(value)
                    elif item_type == 'organ_system':
                        col_name = f"organ_system_{item_id}_{signal_name}"
                        if item_id in snapshot.get('organ_systems', {}):
                            value = snapshot['organ_systems'][item_id].get(signal_name, np.nan)
                            data[col_name].append(value)
                        else:
                            data[col_name].append(np.nan)
                    else:
                        col_name = f"{item_type}_{item_id}_{signal_name}"
                        source = snapshot.get(f"{item_type}s", {})  # entities, organs, tissues
                        if item_id in source:
                            value = source[item_id].get(signal_name, np.nan)
                            data[col_name].append(value)
                        else:
                            data[col_name].append(np.nan)
        
        return pd.DataFrame(data)
    
    @staticmethod
    def get_signal_timeseries(state, item_type, item_id, signal_name):
        """
        Extract single signal as arrays
        
        Args:
            state: SimulationState with history
            item_type: 'entity', 'organ', 'tissue', 'organ_system', or 'organism'
            item_id: ID of the item (None for organism)
            signal_name: Name of the signal
        
        Returns:
            (times, values): Numpy arrays
        
        Example:
            times, glucose = HistoryAnalyzer.get_signal_timeseries(
                state, 'entity', 'blood', 'glucose')
            
            times, epo = HistoryAnalyzer.get_signal_timeseries(
                state, 'organ', 'kidney', 'epo_production_capacity')
            
            times, age = HistoryAnalyzer.get_signal_timeseries(
                state, 'organism', None, 'age')
        """
        times = []
        values = []
        
        for snapshot in state.history:
            times.append(snapshot['time'])
            
            if item_type == 'entity':
                if item_id in snapshot.get('entities', {}):
                    values.append(snapshot['entities'][item_id].get(signal_name, np.nan))
                else:
                    values.append(np.nan)
            
            elif item_type == 'organ':
                if item_id in snapshot.get('organs', {}):
                    values.append(snapshot['organs'][item_id].get(signal_name, np.nan))
                else:
                    values.append(np.nan)
            
            elif item_type == 'tissue':
                if item_id in snapshot.get('tissues', {}):
                    values.append(snapshot['tissues'][item_id].get(signal_name, np.nan))
                else:
                    values.append(np.nan)
            
            elif item_type == 'organ_system':
                if item_id in snapshot.get('organ_systems', {}):
                    values.append(snapshot['organ_systems'][item_id].get(signal_name, np.nan))
                else:
                    values.append(np.nan)
            
            elif item_type == 'organism':
                values.append(snapshot.get('organism', {}).get(signal_name, np.nan))
        
        return np.array(times), np.array(values)
    
    @staticmethod
    def summary_stats(state, item_type, item_id, signal_name):
        """
        Get summary statistics for a signal
        
        Args:
            state: SimulationState with history
            item_type: 'entity', 'organ', 'tissue', or 'organism'
            item_id: ID of the item (None for organism)
            signal_name: Name of the signal
        
        Returns:
            dict: Statistics (mean, std, min, max, final)
        
        Example:
            stats = HistoryAnalyzer.summary_stats(state, 'entity', 'blood', 'glucose')
            print(f"Mean glucose: {stats['mean']:.1f} mg/dL")
        """
        times, values = HistoryAnalyzer.get_signal_timeseries(
            state, item_type, item_id, signal_name)
        
        return {
            'mean': np.nanmean(values),
            'std': np.nanstd(values),
            'min': np.nanmin(values),
            'max': np.nanmax(values),
            'final': values[-1] if len(values) > 0 else np.nan
        }


# Convenience functions
def save_simulation(state, base_filename, formats=['json', 'csv']):
    """
    Save simulation in multiple formats
    
    Args:
        state: SimulationState
        base_filename: Base name (e.g., "my_sim")
        formats: List of formats ['json', 'csv', 'pickle']
    """
    base_path = Path(base_filename)
    
    if 'json' in formats:
        HistoryLogger.save_json(state, f"{base_path}.json")
    
    if 'csv' in formats:
        HistoryLogger.save_csv(state, f"{base_path}.csv")
    
    if 'pickle' in formats:
        HistoryLogger.save_pickle(state, f"{base_path}.pkl")


def quick_plot(state, item_type, item_id, signal_name, save_path=None):
    """
    Quick plot of a single signal
    
    Args:
        state: SimulationState with history
        item_type: 'entity', 'organ', 'tissue', or 'organism'
        item_id: ID of the item (None for organism)
        signal_name: Name of the signal
        save_path: Where to save plot (optional)
    
    Example:
        quick_plot(state, 'entity', 'blood', 'glucose', 'glucose.png')
        quick_plot(state, 'organ', 'kidney', 'epo_production_capacity')
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib not installed. Install with: pip install matplotlib")
        return
    
    times, values = HistoryAnalyzer.get_signal_timeseries(
        state, item_type, item_id, signal_name)
    
    plt.figure(figsize=(10, 6))
    plt.plot(times / 3600, values, linewidth=2)
    plt.xlabel('Time (hours)', fontsize=12)
    
    # Build label
    if item_type == 'organism':
        label = f'organism.{signal_name}'
    else:
        label = f'{item_type}.{item_id}.{signal_name}'
    
    plt.ylabel(label, fontsize=12)
    plt.title(f'{label} over time', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")
    else:
        plt.show()


# Example usage
if __name__ == "__main__":
    print("History logging utilities")
    print()
    print("Usage:")
    print()
    print("# After running simulation:")
    print("from utils.history import save_simulation, quick_plot")
    print()
    print("# Save in multiple formats:")
    print("save_simulation(state, 'results/my_simulation')")
    print("# Creates: my_simulation.json, my_simulation.csv")
    print()
    print("# Quick plots:")
    print("quick_plot(state, 'entity', 'blood', 'glucose', 'results/glucose.png')")
    print("quick_plot(state, 'organ', 'kidney', 'epo_production_capacity', 'results/epo.png')")
    print()
    print("# Save CSV with specific signals:")
    print("signals = [")
    print("    ('entity', 'blood', 'glucose'),")
    print("    ('organ', 'kidney', 'epo_production_capacity'),")
    print("    ('organism', None, 'age')")
    print("]")
    print("HistoryLogger.save_csv(state, 'custom.csv', signals=signals)")
    print()
    print("# Load later:")
    print("from utils.history import HistoryLogger")
    print("history = HistoryLogger.load_json('results/my_simulation.json')")