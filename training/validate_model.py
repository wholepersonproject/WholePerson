import numpy as np
import sys
import io

class SimulationValidator:
    def __init__(self, engine, patient_data):
        self.engine = engine
        self.patient_data = patient_data
    
    def validate_glucose(self):
        print("="*70)
        print("GLUCOSE VALIDATION")
        print("="*70)
        
        cgm_data = self.patient_data['cgm']
        
        # Run simulation silently
        duration = cgm_data['time'][-1] - cgm_data['time'][0]
        self.engine.state.history = []
        self.engine.state.time = 0.0
        
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            self.engine.run(duration_seconds=duration, global_dt=60.0, record_interval=300)
        finally:
            sys.stdout = old_stdout
        
        # Extract simulated
        simulated = [s['entities']['blood']['glucose'] for s in self.engine.state.history]
        simulated = np.array(simulated)
        
        # Interpolate
        sim_times = np.linspace(0, duration, len(simulated))
        sim_interp = np.interp(cgm_data['time'], sim_times, simulated)
        
        # Metrics
        rmse = np.sqrt(np.mean((sim_interp - cgm_data['glucose'])**2))
        mae = np.mean(np.abs(sim_interp - cgm_data['glucose']))
        corr = np.corrcoef(sim_interp, cgm_data['glucose'])[0, 1]
        
        print(f"  RMSE:        {rmse:.2f} mg/dL")
        print(f"  MAE:         {mae:.2f} mg/dL")
        print(f"  Correlation: {corr:.3f}")
        print("="*70)
        print()
        
        return {'rmse': rmse, 'mae': mae, 'correlation': corr}

