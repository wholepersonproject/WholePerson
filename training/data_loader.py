import pandas as pd
import numpy as np
from pathlib import Path

class PatientDataLoader:
    def __init__(self, patient_id, data_dir="data"):
        self.patient_id = patient_id
        self.data_dir = Path(data_dir)
        self.data = {}
    
    def load_cgm(self, filename=None):
        if filename is None:
            filename = self.data_dir / f"{self.patient_id}_cgm.csv"
        
        # Create synthetic data if file doesn't exist
        if not filename.exists():
            print(f"  Creating synthetic CGM data for {self.patient_id}")
            self._create_synthetic_cgm(filename)
        
        df = pd.read_csv(filename)
        self.data['cgm'] = {
            'time': df['timestamp'].values,
            'glucose': df['glucose_mg_dl'].values
        }
        print(f"✓ Loaded CGM: {len(self.data['cgm']['glucose'])} readings")
        return self.data['cgm']
    
    def _create_synthetic_cgm(self, filename):
        """Create synthetic CGM data for demonstration"""
        # 24 hours at 5-minute intervals
        times = np.arange(0, 24*3600, 300)
        
        # Baseline glucose with circadian rhythm
        baseline = 90 + 10 * np.sin(times / 3600 * 2 * np.pi / 24)
        
        # Add meal spikes
        glucose = baseline.copy()
        
        # Breakfast at 8h
        breakfast_time = 8 * 3600
        breakfast_mask = (times >= breakfast_time) & (times < breakfast_time + 7200)
        time_since_meal = (times[breakfast_mask] - breakfast_time) / 3600
        glucose[breakfast_mask] += 40 * np.exp(-time_since_meal / 1.5)
        
        # Lunch at 12h
        lunch_time = 12 * 3600
        lunch_mask = (times >= lunch_time) & (times < lunch_time + 7200)
        time_since_meal = (times[lunch_mask] - lunch_time) / 3600
        glucose[lunch_mask] += 50 * np.exp(-time_since_meal / 1.5)
        
        # Dinner at 18h
        dinner_time = 18 * 3600
        dinner_mask = (times >= dinner_time) & (times < dinner_time + 7200)
        time_since_meal = (times[dinner_mask] - dinner_time) / 3600
        glucose[dinner_mask] += 45 * np.exp(-time_since_meal / 1.5)
        
        # Add noise
        glucose += np.random.normal(0, 3, len(glucose))
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': times,
            'glucose_mg_dl': glucose
        })
        
        # Save to file
        filename.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filename, index=False)
    
    def load_all(self):
        self.load_cgm()
        return self.data