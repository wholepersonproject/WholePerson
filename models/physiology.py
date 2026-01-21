import numpy as np
from models.base import ProcessModel, TimeScale

class InsulinSecretion(ProcessModel):
    def __init__(self, glucose_sensitivity=1.0, basal_secretion=5.0, max_secretion=150.0):
        super().__init__("insulin_secretion", TimeScale.MINUTES)
        self.glucose_sensitivity = glucose_sensitivity
        self.basal_secretion = basal_secretion
        self.max_secretion = max_secretion
    
    def step(self, state, dt):
        glucose = state.get_signal('pancreas', 'glucose')
        
        K = 90.0
        n = 1.7
        stimulated = (self.max_secretion - self.basal_secretion) * \
                     (glucose**n) / (K**n + glucose**n)
        secretion_rate = self.basal_secretion + stimulated * self.glucose_sensitivity
        
        state.update_signal('blood', 'insulin', secretion_rate * (dt/60.0))
        state.update_signal('pancreas', 'insulin', secretion_rate * 1.5 * (dt/60.0))


class GlucagonSecretion(ProcessModel):
    def __init__(self, basal_secretion=60.0):
        super().__init__("glucagon_secretion", TimeScale.MINUTES)
        self.basal_secretion = basal_secretion
    
    def step(self, state, dt):
        glucose = state.get_signal('blood', 'glucose')
        insulin = state.get_signal('blood', 'insulin')
        
        glucose_factor = 70.0 / (glucose + 1.0)
        insulin_inhibition = 1.0 / (1.0 + insulin / 5.0)
        secretion_rate = self.basal_secretion * glucose_factor * insulin_inhibition
        
        state.update_signal('blood', 'glucagon', secretion_rate * (dt/60.0))
        state.update_signal('pancreas', 'glucagon', secretion_rate * 1.2 * (dt/60.0))


class GlucoseUptake(ProcessModel):
    def __init__(self, target_entity='muscle_tissue', basal_rate=0.1, insulin_sensitivity=1.0):
        super().__init__(f"glucose_uptake_{target_entity}", TimeScale.MINUTES)
        self.target_entity = target_entity
        self.basal_rate = basal_rate
        self.insulin_sensitivity = insulin_sensitivity
    
    def step(self, state, dt):
        blood_glucose = state.get_signal('blood', 'glucose')
        blood_insulin = state.get_signal('blood', 'insulin')
        
        insulin_factor = 1.0 + (blood_insulin / 5.0) * self.insulin_sensitivity
        uptake_rate = self.basal_rate * insulin_factor * (blood_glucose / 90.0)
        
        amount = uptake_rate * (dt / 60.0)
        state.update_signal('blood', 'glucose', -amount)
        state.update_signal(self.target_entity, 'glucose', amount)


class GlycogenSynthesis(ProcessModel):
    def __init__(self):
        super().__init__("glycogen_synthesis", TimeScale.HOURS)
    
    def step(self, state, dt):
        fed_status = state.get_organism_state('fed_status', 'fasted')
        
        if fed_status == 'fed':
            liver_glucose = state.get_signal('liver', 'glucose')
            liver_insulin = state.get_signal('liver', 'insulin')
            
            if liver_insulin > 8.0:
                synthesis = 0.05 * liver_glucose * (liver_insulin / 10.0)
                amount = synthesis * (dt / 3600.0)
                state.update_signal('liver', 'glucose', -amount)
                state.update_signal('liver', 'glycogen', amount)
            
            muscle_glucose = state.get_signal('muscle_tissue', 'glucose')
            muscle_insulin = state.get_signal('muscle_tissue', 'insulin')
            
            if muscle_insulin > 8.0:
                synthesis = 0.03 * muscle_glucose * (muscle_insulin / 10.0)
                amount = synthesis * (dt / 3600.0)
                state.update_signal('muscle_tissue', 'glucose', -amount)
                state.update_signal('muscle_tissue', 'glycogen', amount)


class GlycogenBreakdown(ProcessModel):
    def __init__(self):
        super().__init__("glycogen_breakdown", TimeScale.HOURS)
    
    def step(self, state, dt):
        fed_status = state.get_organism_state('fed_status', 'fasted')
        glucagon = state.get_signal('liver', 'glucagon')
        
        if fed_status == 'fasted' and glucagon > 70.0:
            liver_glycogen = state.get_signal('liver', 'glycogen')
            breakdown_rate = 0.03 * (glucagon / 80.0)
            amount = min(breakdown_rate * (dt / 3600.0), liver_glycogen)
            
            state.update_signal('liver', 'glycogen', -amount)
            state.update_signal('liver', 'glucose', amount)
            state.update_signal('blood', 'glucose', amount * 0.5)


class HepaticGlucoseProduction(ProcessModel):
    def __init__(self, production_rate=2.0):
        super().__init__("hepatic_glucose_production", TimeScale.HOURS)
        self.production_rate = production_rate
    
    def step(self, state, dt):
        glucagon = state.get_signal('liver', 'glucagon')
        insulin = state.get_signal('liver', 'insulin')
        fed_status = state.get_organism_state('fed_status', 'fasted')
        
        # Base production
        glucagon_factor = glucagon / 60.0
        insulin_factor = 1.0 / (1.0 + insulin / 10.0)
        
        # Increase production in fasted state
        fasting_boost = 1.5 if fed_status == 'fasted' else 0.5
        
        production = self.production_rate * glucagon_factor * insulin_factor * fasting_boost
        amount = production * (dt / 3600.0)
        
        state.update_signal('liver', 'glucose', amount)
        state.update_signal('blood', 'glucose', amount * 0.8)


class CirculatoryTransport(ProcessModel):
    def __init__(self):
        super().__init__("circulatory_transport", TimeScale.SECONDS)
    
    def step(self, state, dt):
        for flow_id, flow in state.flows.items():
            from_entity = flow['from']
            to_entity = flow['to']
            rate = flow['rate']
            
            if from_entity not in state.entities or to_entity not in state.entities:
                continue
            
            from_ent = state.entities[from_entity]
            to_ent = state.entities[to_entity]
            
            if from_ent['representation'] == 'lumped':
                from_signals = set(from_ent['signals'].keys())
            elif from_ent['representation'] == 'spatial':
                from_signals = set(from_ent['signals'].keys())
            else:
                continue
            
            if to_ent['representation'] == 'lumped':
                to_signals = set(to_ent['signals'].keys())
            elif to_ent['representation'] == 'spatial':
                to_signals = set(to_ent['signals'].keys())
            else:
                continue
            
            shared = from_signals.intersection(to_signals)
            
            for signal in shared:
                if signal in ['glycogen']:
                    continue
                
                C_from = state.get_signal(from_entity, signal)
                C_to = state.get_signal(to_entity, signal)
                
                volume = to_ent.get('volume', 1.0)
                transport_rate = (rate / volume) * (C_from - C_to)
                amount = transport_rate * (dt / 60.0)
                
                state.update_signal(to_entity, signal, amount)


class HormoneDegradation(ProcessModel):
    def __init__(self):
        super().__init__("hormone_degradation", TimeScale.MINUTES)
        self.decay_rates = {
            'insulin': 0.15,  # Increased from 0.05
            'glucagon': 0.08  # Increased from 0.03
        }
    
    def step(self, state, dt):
        for entity_id, entity in state.entities.items():
            rep = entity['representation']
            
            if rep in ['lumped', 'spatial']:
                for signal_name in list(entity.get('signals', {}).keys()):
                    if signal_name in self.decay_rates:
                        rate = self.decay_rates[signal_name]
                        current = state.get_signal(entity_id, signal_name)
                        decay = current * rate * (dt / 60.0)
                        state.update_signal(entity_id, signal_name, -decay)

class f_cell_polypeptide_0036322(ProcessModel):  
    """
    Simple F cell agent model
    
    Each F cell:
    - Has individual secretion capacity
    - Responds to meal signals
    - Can become exhausted with overuse
    """
    
    def __init__(self):
        super().__init__("f_cell_dynamics", TimeScale.MINUTES)
    
    def step(self, state, dt):
        # Check if F cells exist
        if 'f_cells' not in state.entities:
            return
        
        # Get meal status
        fed_status = state.get_organism_state('fed_status', 'fasted')
        
        # Get current PP level
        pp_level = state.get_signal('blood', 'pancreatic_polypeptide')
        if pp_level is None or pp_level == 0:
            pp_level = 30.0
        
        # Process each F cell
        total_secretion = 0.0
        
        for agent in state.entities['f_cells']['agents']:
            # Each cell has individual capacity
            # Agent properties are in agent['state']
            secretion_capacity = agent['state']['secretion_capacity']
            activation_threshold = agent['state']['activation_threshold']
            
            # Cells secrete if fed and not exhausted
            if fed_status == 'fed':
                # Random variation in response
                if np.random.random() < activation_threshold:
                    # This cell secretes
                    amount = secretion_capacity * (dt / 60)  # pg/min
                    total_secretion += amount
                    
                    # Using the cell fatigues it slightly
                    agent['state']['secretion_capacity'] *= 0.9999
            else:
                # Fasted: cells recover capacity
                if agent['state']['secretion_capacity'] < agent['state'].get('max_capacity', secretion_capacity):
                    agent['state']['secretion_capacity'] *= 1.0001
        
        # Update blood PP level
        # Secretion increases, clearance decreases
        clearance = pp_level * 0.12 * (dt / 60)  # Half-life ~6 min
        net_change = total_secretion - clearance
        
        new_pp = max(10, pp_level + net_change)
        state.set_signal('blood', 'pancreatic_polypeptide', new_pp)