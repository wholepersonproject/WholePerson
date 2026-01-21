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
                'organism': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                           for k, v in snapshot['organism'].items()}
            }
            
            for entity_id, signals in snapshot['entities'].items():
                snap_copy['entities'][entity_id] = {}
                for signal_name, value in signals.items():
                    if isinstance(value, np.ndarray):
                        snap_copy['entities'][entity_id][signal_name] = value.tolist()
                    elif isinstance(value, (np.floating, np.integer)):
                        snap_copy['entities'][entity_id][signal_name] = float(value)
                    else:
                        snap_copy['entities'][entity_id][signal_name] = value
            
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
            signals: List of (entity, signal) tuples to save
                    If None, saves common ones
        
        Example:
            signals = [
                ('blood', 'glucose'),
                ('blood', 'insulin'),
                ('liver', 'glycogen')
            ]
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Default signals if not specified
        if signals is None:
            signals = [
                ('blood', 'glucose'),
                ('blood', 'insulin'),
                ('blood', 'glucagon'),
                ('liver', 'glycogen')
            ]
        
        # Build CSV manually (no pandas dependency)
        lines = []
        
        # Header
        header = ['time_seconds', 'time_hours']
        for entity_id, signal_name in signals:
            header.append(f"{entity_id}.{signal_name}")
        lines.append(','.join(header))
        
        # Data rows
        for snapshot in state.history:
            row = [
                str(snapshot['time']),
                str(snapshot['time'] / 3600)
            ]
            
            for entity_id, signal_name in signals:
                if entity_id in snapshot['entities']:
                    if signal_name in snapshot['entities'][entity_id]:
                        value = snapshot['entities'][entity_id][signal_name]
                        row.append(str(float(value)))
                    else:
                        row.append('')
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
        """Convert history to pandas DataFrame (requires pandas)"""
        try:
            import pandas as pd
        except ImportError:
            print("ERROR: pandas not installed. Install with: pip install pandas")
            return None
        
        if signals is None:
            signals = [
                ('blood', 'glucose'),
                ('blood', 'insulin'),
                ('blood', 'glucagon'),
                ('liver', 'glycogen')
            ]
        
        data = {'time': []}
        for entity_id, signal_name in signals:
            data[f"{entity_id}_{signal_name}"] = []
        
        for snapshot in state.history:
            data['time'].append(snapshot['time'] / 3600)  # hours
            
            for entity_id, signal_name in signals:
                if entity_id in snapshot['entities']:
                    value = snapshot['entities'][entity_id].get(signal_name, np.nan)
                    data[f"{entity_id}_{signal_name}"].append(value)
                else:
                    data[f"{entity_id}_{signal_name}"].append(np.nan)
        
        return pd.DataFrame(data)
    
    @staticmethod
    def get_signal_timeseries(state, entity_id, signal_name):
        """Extract single signal as arrays"""
        times = []
        values = []
        
        for snapshot in state.history:
            times.append(snapshot['time'])
            if entity_id in snapshot['entities']:
                values.append(snapshot['entities'][entity_id].get(signal_name, np.nan))
            else:
                values.append(np.nan)
        
        return np.array(times), np.array(values)
    
    @staticmethod
    def summary_stats(state, entity_id, signal_name):
        """Get summary statistics for a signal"""
        times, values = HistoryAnalyzer.get_signal_timeseries(state, entity_id, signal_name)
        
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


def quick_plot(state, entity_id, signal_name, save_path=None):
    """Quick plot of a single signal"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib not installed. Install with: pip install matplotlib")
        return
    
    times, values = HistoryAnalyzer.get_signal_timeseries(state, entity_id, signal_name)
    
    plt.figure(figsize=(10, 6))
    plt.plot(times / 3600, values, linewidth=2)
    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel(f'{entity_id}.{signal_name}', fontsize=12)
    plt.title(f'{entity_id}.{signal_name} over time', fontsize=14)
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
    print("# Quick plot:")
    print("quick_plot(state, 'blood', 'glucose', 'results/glucose_plot.png')")
    print()
    print("# Load later:")
    print("from utils.history import HistoryLogger")
    print("history = HistoryLogger.load_json('results/my_simulation.json')")