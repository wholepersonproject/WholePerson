import numpy as np
import yaml

class Perturbation:
    def __init__(self, perturbation_id, config):
        self.perturbation_id = perturbation_id
        self.config = config
        self.start_time = None
        self.end_time = None
        self.active = False
    
    def activate(self, time):
        self.start_time = time
        duration = self.config.get('parameters', {}).get('duration_minutes', 60)
        self.end_time = time + duration * 60
        self.active = True
    
    def is_active(self, time):
        if not self.active:
            return False
        if self.end_time and time >= self.end_time:
            self.active = False
            return False
        return True
    
    def apply(self, state, dt):
        if not self.is_active(state.time):
            return
        
        for effect in self.config.get('effects', []):
            self._apply_effect(state, effect, dt)
    
    def _apply_effect(self, state, effect, dt):
        target = effect['target']
        
        # Handle model parameter modification
        if target.startswith('model.'):
            self._apply_model_effect(state, target, effect, dt)
        
        # Handle organism state
        elif target.startswith('organism.'):
            state_name = target.split('.')[1]
            if 'value' in effect:
                state.set_organism_state(state_name, effect['value'])
        
        # Handle entity signal
        elif '.' in target:
            entity_id, signal_name = target.split('.')
            
            if 'profile' in effect:
                if effect['profile'] == 'spike':
                    self._apply_spike(state, entity_id, signal_name, effect, dt)
                elif effect['profile'] == 'injection':
                    self._apply_injection(state, entity_id, signal_name, effect, dt)
            elif 'rate' in effect:
                state.update_signal(entity_id, signal_name, effect['rate'] * dt)
            elif 'factor' in effect:
                current = state.get_signal(entity_id, signal_name)
                state.set_signal(entity_id, signal_name, current * effect['factor'])
    
    def _apply_model_effect(self, state, target, effect, dt):
        '''Modify model parameters (e.g., aging, chronic conditions)'''
        # Parse: "model.insulin_secretion.glucose_sensitivity"
        parts = target.split('.')
        if len(parts) != 3:
            return
        
        _, process_id, param_name = parts
        
        # Access model via engine
        if not state._engine or process_id not in state._engine.models:
            return
        
        model = state._engine.models[process_id]
        
        if not hasattr(model, param_name):
            return
        
        if 'rate' in effect:
            # Gradual change: param += rate * dt
            current = getattr(model, param_name)
            setattr(model, param_name, current + effect['rate'] * dt)
        
        elif 'factor' in effect:
            # Multiplicative: param *= factor
            current = getattr(model, param_name)
            setattr(model, param_name, current * effect['factor'])
    
    def _apply_spike(self, state, entity_id, signal_name, effect, dt):
        time_since_start = state.time - self.start_time
        peak_time = effect['peak_time'] * 60
        peak_magnitude = effect['peak_magnitude']
        decay_rate = effect['decay_rate']
        
        if time_since_start < peak_time:
            progress = time_since_start / peak_time
            target_value = peak_magnitude * progress
        else:
            time_since_peak = time_since_start - peak_time
            target_value = peak_magnitude * np.exp(-decay_rate * time_since_peak / 60)
        
        current = state.get_signal(entity_id, signal_name)
        alpha = 0.1
        new_value = current * (1 - alpha) + target_value * alpha
        state.set_signal(entity_id, signal_name, new_value)
    
    def _apply_injection(self, state, entity_id, signal_name, effect, dt):
        time_since_start = state.time - self.start_time
        peak_time = effect['peak_time'] * 60
        peak_magnitude = effect['peak_magnitude']
        decay_rate = effect['decay_rate']
        
        if time_since_start < peak_time:
            progress = time_since_start / peak_time
            bolus = peak_magnitude * np.sin(progress * np.pi / 2)
        else:
            time_since_peak = time_since_start - peak_time
            bolus = peak_magnitude * np.exp(-decay_rate * time_since_peak / 60)
        
        state.update_signal(entity_id, signal_name, bolus * (dt / 60.0))


class PerturbationManager:
    def __init__(self, config_path="configs/perturbations.yaml"):
        with open(config_path, 'r') as f:
            self.library = yaml.safe_load(f)
        self.perturbations = []
    
    def add_perturbation(self, category, name, start_time):
        if category not in self.library:
            raise ValueError(f"Unknown category: {category}")
        if name not in self.library[category]:
            raise ValueError(f"Unknown perturbation: {name}")
        
        config = self.library[category][name]
        perturb = Perturbation(f"{category}_{name}", config)
        perturb.activate(start_time)
        self.perturbations.append(perturb)
        print(f"  Added {category}/{name} at t={start_time/3600:.1f}h")
        return perturb
    
    def apply_all(self, state, dt):
        for perturb in self.perturbations:
            perturb.apply(state, dt)
        self.perturbations = [p for p in self.perturbations if p.active]