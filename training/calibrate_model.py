import numpy as np
from scipy.optimize import minimize

class ModelCalibrator:
    def __init__(self, engine, patient_data):
        self.engine = engine
        self.patient_data = patient_data
        self.best_params = None
    
    def calibrate(self, target_params, method='L-BFGS-B', max_iter=20):
        '''
        Calibrate model parameters to patient data
        
        Args:
            target_params: dict of {process_id: [param_names]}
            method: optimization method
            max_iter: maximum iterations
        
        Example:
            target_params = {
                'insulin_secretion': ['glucose_sensitivity', 'max_secretion'],
                'glucose_uptake_muscle': ['insulin_sensitivity']
            }
        '''
        print("="*70)
        print("MODEL CALIBRATION")
        print("="*70)
        
        # Extract initial parameters
        initial_params = []
        param_map = []
        
        for process_id, param_names in target_params.items():
            if process_id not in self.engine.models:
                continue
            model = self.engine.models[process_id]
            for param_name in param_names:
                if hasattr(model, param_name):
                    initial_params.append(getattr(model, param_name))
                    param_map.append((process_id, param_name))
        
        initial_params = np.array(initial_params)
        
        print(f"Calibrating {len(initial_params)} parameters:")
        for process_id, param_name in param_map:
            model = self.engine.models[process_id]
            print(f"  {process_id}.{param_name} = {getattr(model, param_name):.3f}")
        print()
        
        # Define loss function
        def loss_fn(params):
            # Set parameters
            for i, (process_id, param_name) in enumerate(param_map):
                setattr(self.engine.models[process_id], param_name, params[i])
            
            # Reset and run
            self.engine.state.time = 0.0
            self.engine.state.history = []
            
            cgm_data = self.patient_data['cgm']
            duration = cgm_data['time'][-1] - cgm_data['time'][0]
            
            # Silent run
            import sys
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
                self.engine.run(duration_seconds=duration, global_dt=60.0, record_interval=300)
            finally:
                sys.stdout = old_stdout
            
            # Extract simulated glucose
            simulated = [s['entities']['blood']['glucose'] for s in self.engine.state.history]
            simulated = np.array(simulated)
            
            # Interpolate to match patient data
            sim_times = np.linspace(0, duration, len(simulated))
            sim_interp = np.interp(cgm_data['time'], sim_times, simulated)
            
            mse = np.mean((sim_interp - cgm_data['glucose'])**2)
            return mse
        
        # Optimize
        bounds = [(p * 0.1, p * 10.0) for p in initial_params]
        
        result = minimize(
            loss_fn,
            initial_params,
            method=method,
            bounds=bounds,
            options={'maxiter': max_iter}
        )
        
        self.best_params = result.x
        
        # Set best parameters
        for i, (process_id, param_name) in enumerate(param_map):
            setattr(self.engine.models[process_id], param_name, result.x[i])
        
        print()
        print("="*70)
        print("CALIBRATION COMPLETE")
        print("="*70)
        print(f"Final loss: {result.fun:.4f}")
        print()
        print("Optimized parameters:")
        for i, (process_id, param_name) in enumerate(param_map):
            old_val = initial_params[i]
            new_val = result.x[i]
            change = ((new_val - old_val) / old_val) * 100
            print(f"  {process_id}.{param_name}: {old_val:.3f} → {new_val:.3f} ({change:+.1f}%)")
        print("="*70)
        print()
        
        return self.best_params