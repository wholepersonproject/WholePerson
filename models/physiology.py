import numpy as np
from models.base import ProcessModel, TimeScale

class InsulinSecretion(ProcessModel):
    inputs = {
        'glucose': ('pancreas', 'glucose')
    }
    outputs = {
        'blood_insulin': ('blood', 'insulin'),
        'pancreas_insulin': ('pancreas', 'insulin')
    }
    
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
    inputs = {
        'glucose': ('blood', 'glucose'),
        'insulin': ('blood', 'insulin')
    }
    outputs = {
        'blood_glucagon': ('blood', 'glucagon'),
        'pancreas_glucagon': ('pancreas', 'glucagon')
    }
    
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
    # Note: inputs/outputs set in __init__ because target_entity varies
    
    def __init__(self, target_entity='muscle_tissue', basal_rate=0.1, insulin_sensitivity=1.0):
        super().__init__(f"glucose_uptake_{target_entity}", TimeScale.MINUTES)
        self.target_entity = target_entity
        self.basal_rate = basal_rate
        self.insulin_sensitivity = insulin_sensitivity
        
        # Set inputs/outputs based on target
        self.inputs = {
            'blood_glucose': ('blood', 'glucose'),
            'blood_insulin': ('blood', 'insulin')
        }
        self.outputs = {
            'blood_glucose': ('blood', 'glucose'),
            'target_glucose': (target_entity, 'glucose')
        }
    
    def step(self, state, dt):
        blood_glucose = state.get_signal('blood', 'glucose')
        blood_insulin = state.get_signal('blood', 'insulin')
        
        insulin_factor = 1.0 + (blood_insulin / 5.0) * self.insulin_sensitivity
        uptake_rate = self.basal_rate * insulin_factor * (blood_glucose / 90.0)
        
        amount = uptake_rate * (dt / 60.0)
        state.update_signal('blood', 'glucose', -amount)
        state.update_signal(self.target_entity, 'glucose', amount)


class GlycogenSynthesis(ProcessModel):
    inputs = {
        'fed_status': ('fed_status', None, 'organism'),
        'liver_glucose': ('liver', 'glucose'),
        'liver_insulin': ('liver', 'insulin'),
        'muscle_glucose': ('muscle_tissue', 'glucose'),
        'muscle_insulin': ('muscle_tissue', 'insulin')
    }
    outputs = {
        'liver_glucose': ('liver', 'glucose'),
        'liver_glycogen': ('liver', 'glycogen'),
        'muscle_glucose': ('muscle_tissue', 'glucose'),
        'muscle_glycogen': ('muscle_tissue', 'glycogen')
    }
    
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
    inputs = {
        'fed_status': ('fed_status', None, 'organism'),
        'glucagon': ('liver', 'glucagon'),
        'glycogen': ('liver', 'glycogen')
    }
    outputs = {
        'liver_glycogen': ('liver', 'glycogen'),
        'liver_glucose': ('liver', 'glucose'),
        'blood_glucose': ('blood', 'glucose')
    }
    
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
    inputs = {
        'glucagon': ('liver', 'glucagon'),
        'insulin': ('liver', 'insulin'),
        'fed_status': ('fed_status', None, 'organism')
    }
    outputs = {
        'liver_glucose': ('liver', 'glucose'),
        'blood_glucose': ('blood', 'glucose')
    }
    
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
    # Note: This process operates on all flows defined in state.flows
    # Inputs/outputs are dynamic based on shared signals between flow endpoints
    inputs = {}  # Dynamic - reads from all entities in flows
    outputs = {}  # Dynamic - writes to all entities in flows
    
    def __init__(self):
        super().__init__("circulatory_transport", TimeScale.SECONDS)
    
    def step(self, state, dt):
        for flow_id, flow in state.flows.items():
            from_id = flow['from']
            to_id = flow['to']
            rate = flow['rate']
            
            # Helper function to get item (entity, organ, or tissue)
            def get_item(id):
                if id in state.entities:
                    return state.entities[id]
                elif id in state.organs:
                    return state.organs[id]
                elif id in state.tissues:
                    return state.tissues[id]
                return None
            
            from_item = get_item(from_id)
            to_item = get_item(to_id)
            
            if not from_item or not to_item:
                continue
            
            # Get signals from source
            from_rep = from_item.get('representation', 'lumped')
            if from_rep in ['lumped', 'spatial']:
                from_signals = set(from_item['signals'].keys())
            else:
                continue
            
            # Get signals from target
            to_rep = to_item.get('representation', 'lumped')
            if to_rep in ['lumped', 'spatial']:
                to_signals = set(to_item['signals'].keys())
            else:
                continue
            
            # Transport shared signals
            shared = from_signals.intersection(to_signals)
            
            for signal in shared:
                if signal in ['glycogen']:
                    continue
                
                C_from = state.get_signal(from_id, signal)
                C_to = state.get_signal(to_id, signal)
                
                if C_from is None or C_to is None:
                    continue
                
                volume = to_item.get('volume', 1.0)
                transport_rate = (rate / volume) * (C_from - C_to)
                amount = transport_rate * (dt / 60.0)
                
                state.update_signal(to_id, signal, amount)


class HormoneDegradation(ProcessModel):
    # Note: Operates on all entities with insulin/glucagon signals
    inputs = {}  # Dynamic - reads insulin/glucagon from all entities
    outputs = {}  # Dynamic - writes to insulin/glucagon in all entities
    
    def __init__(self):
        super().__init__("hormone_degradation", TimeScale.MINUTES)
        self.decay_rates = {
            'insulin': 0.15,           # Half-life ~5 min
            'glucagon': 0.08,          # Half-life ~9 min
            'erythropoietin': 0.002    # Half-life ~5 hours (slow clearance)
        }
    
    def step(self, state, dt):
        # Degrade hormones in entities
        for entity_id, entity in state.entities.items():
            rep = entity['representation']
            
            if rep in ['lumped', 'spatial']:
                for signal_name in list(entity.get('signals', {}).keys()):
                    if signal_name in self.decay_rates:
                        rate = self.decay_rates[signal_name]
                        current = state.get_signal(entity_id, signal_name)
                        if current is not None and current > 0:
                            decay = current * rate * (dt / 60.0)
                            state.update_signal(entity_id, signal_name, -decay)
        
        # Degrade hormones in organs
        for organ_id, organ in state.organs.items():
            for signal_name in list(organ.get('signals', {}).keys()):
                if signal_name in self.decay_rates:
                    rate = self.decay_rates[signal_name]
                    current = state.get_signal(organ_id, signal_name)
                    if current is not None and current > 0:
                        decay = current * rate * (dt / 60.0)
                        state.update_signal(organ_id, signal_name, -decay)
        
        # Degrade hormones in tissues
        for tissue_id, tissue in state.tissues.items():
            for signal_name in list(tissue.get('signals', {}).keys()):
                if signal_name in self.decay_rates:
                    rate = self.decay_rates[signal_name]
                    current = state.get_signal(tissue_id, signal_name)
                    if current is not None and current > 0:
                        decay = current * rate * (dt / 60.0)
                        state.update_signal(tissue_id, signal_name, -decay)

class f_cell_polypeptide_0036322(ProcessModel):  
    """
    Simple F cell agent model
    
    Each F cell:
    - Has individual secretion capacity
    - Responds to meal signals
    - Can become exhausted with overuse
    """
    
    inputs = {
        'fed_status': ('fed_status', None, 'organism'),
        'pp_level': ('blood', 'pancreatic_polypeptide')
    }
    outputs = {
        'blood_pp': ('blood', 'pancreatic_polypeptide')
    }
    
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

class ErythropoietinProduction(ProcessModel):
    """
    EPO production by kidneys in response to tissue oxygen
    
    Timescale: Hours (EPO synthesis and release)
    Location: Kidney peritubular fibroblasts
    """
    
    inputs = {
        'tissue_O2_saturation': ('tissue_oxygen_saturation', None, 'organism'),
        'epo_capacity': ('kidney', 'epo_production_capacity'),  # Smart lookup finds in organs
        'functional_mass': ('kidney', 'functional_mass')
    }
    outputs = {
        'blood_epo': ('blood', 'erythropoietin')
    }
    
    def __init__(self, basal_epo=10.0, max_epo=200.0):
        super().__init__("epo_production", TimeScale.HOURS)
        self.basal_epo = basal_epo  # mU/mL baseline
        self.max_epo = max_epo      # mU/mL maximum
    
    def step(self, state, dt):
        # Get inputs
        tissue_O2 = state.get_organism_state('tissue_oxygen_saturation', 95.0)
        epo_capacity = state.get_signal('kidney', 'epo_production_capacity')
        if epo_capacity is None:
            epo_capacity = 1.0
        kidney_mass = state.get_signal('kidney', 'functional_mass')
        if kidney_mass is None:
            kidney_mass = 1.0
        
        # EPO production inversely proportional to O2 (hypoxia drives EPO)
        target_O2 = 95.0  # Normal tissue O2 saturation (%)
        hypoxia_factor = max(0.0, (target_O2 - tissue_O2) / target_O2)
        
        # Sigmoidal response to hypoxia
        hill_n = 2.5
        K_half = 0.1  # 10% hypoxia gives half-max response
        epo_production_rate = self.basal_epo + \
            (self.max_epo - self.basal_epo) * \
            (hypoxia_factor**hill_n) / (K_half**hill_n + hypoxia_factor**hill_n)
        
        # Scale by kidney capacity and mass
        epo_production_rate *= epo_capacity * kidney_mass
        
        # Update blood EPO
        epo_amount = epo_production_rate * (dt / 3600.0)
        state.update_signal('blood', 'erythropoietin', epo_amount)


class ErythropoiesisStimulation(ProcessModel):
    """
    Bone marrow RBC production stimulated by EPO
    
    Timescale: Days (cell maturation)
    Location: Bone marrow erythroid progenitors
    """
    
    inputs = {
        'blood_epo': ('blood', 'erythropoietin'),
        'marrow_capacity': ('bone_marrow', 'erythropoiesis_capacity')  # Smart lookup finds in organs
    }
    outputs = {
        'blood_rbc_count': ('blood', 'rbc_count'),
        'blood_hemoglobin': ('blood', 'hemoglobin')
    }
    
    def __init__(self, basal_production=2.5e11):
        super().__init__("erythropoiesis", TimeScale.DAYS)
        self.basal_production = float(basal_production)  # RBCs/day (baseline ~250 billion)
    
    def step(self, state, dt):
        # Get inputs
        epo = state.get_signal('blood', 'erythropoietin')
        if epo is None:
            epo = 10.0  # Basal EPO
        
        marrow_capacity = state.get_signal('bone_marrow', 'erythropoiesis_capacity')
        if marrow_capacity is None:
            marrow_capacity = 1.0
        
        # EPO-stimulated production (sigmoidal)
        epo_baseline = 10.0  # mU/mL
        K_epo = 20.0  # Half-maximal EPO
        hill_n = 2.0
        epo_factor = 1.0 + 3.0 * (epo**hill_n) / (K_epo**hill_n + epo**hill_n)
        
        # Total RBC production
        rbc_production_rate = self.basal_production * epo_factor * marrow_capacity
        
        # Update RBC count (cells/µL)
        # Average adult has ~5L blood, ~5 million RBC/µL
        blood_volume_liters = 5.0
        new_rbcs = rbc_production_rate * (dt / 86400.0)  # RBCs produced this timestep
        rbc_per_uL_increase = new_rbcs / (blood_volume_liters * 1e6)  # Convert to per µL
        
        state.update_signal('blood', 'rbc_count', rbc_per_uL_increase)
        
        # Hemoglobin (g/dL) - each RBC carries ~30 pg Hb
        hb_increase = (new_rbcs * 30e-12) / (blood_volume_liters * 10)  # g/dL (convert L to dL)
        state.update_signal('blood', 'hemoglobin', hb_increase)


class RBCTurnover(ProcessModel):
    """
    RBC removal by spleen (120-day lifespan)
    
    Timescale: Days
    Location: Spleen and liver (reticuloendothelial system)
    """
    
    inputs = {
        'rbc_count': ('blood', 'rbc_count'),
        'hemoglobin': ('blood', 'hemoglobin')
    }
    outputs = {
        'blood_rbc_count': ('blood', 'rbc_count'),
        'blood_hemoglobin': ('blood', 'hemoglobin')
    }
    
    def __init__(self, rbc_lifespan_days=120.0):
        super().__init__("rbc_turnover", TimeScale.DAYS)
        self.rbc_lifespan_days = rbc_lifespan_days
    
    def step(self, state, dt):
        # Get current RBC count and hemoglobin
        rbc_count = state.get_signal('blood', 'rbc_count')
        hemoglobin = state.get_signal('blood', 'hemoglobin')
        
        if rbc_count is None or rbc_count == 0:
            rbc_count = 5.0e6  # Normal ~5 million/µL
        if hemoglobin is None or hemoglobin == 0:
            hemoglobin = 15.0  # Normal ~15 g/dL
        
        # First-order removal (exponential decay)
        removal_rate = 1.0 / self.rbc_lifespan_days  # per day
        
        rbc_removed = rbc_count * removal_rate * (dt / 86400.0)
        hb_removed = hemoglobin * removal_rate * (dt / 86400.0)
        
        state.update_signal('blood', 'rbc_count', -rbc_removed)
        state.update_signal('blood', 'hemoglobin', -hb_removed)


class OxygenDelivery(ProcessModel):
    """
    Oxygen transport from lungs to tissues via hemoglobin
    
    Timescale: Seconds (circulation time)
    Location: Cardiovascular system
    """
    
    inputs = {
        'hemoglobin': ('blood', 'hemoglobin'),
        'rbc_count': ('blood', 'rbc_count'),
        'cardiac_output': ('cardiac_output', None, 'organism')
    }
    outputs = {
        'tissue_O2_saturation': ('tissue_oxygen_saturation', None, 'organism')
    }
    
    def __init__(self):
        super().__init__("oxygen_delivery", TimeScale.SECONDS)
    
    def step(self, state, dt):
        # Get inputs
        hemoglobin = state.get_signal('blood', 'hemoglobin')
        if hemoglobin is None or hemoglobin == 0:
            hemoglobin = 15.0  # g/dL
        
        cardiac_output = state.get_organism_state('cardiac_output', 5.0)  # L/min
        
        # Oxygen carrying capacity
        # Each g of Hb carries ~1.34 mL O2 when saturated
        O2_capacity = hemoglobin * 1.34  # mL O2 per dL blood
        
        # Oxygen delivery rate
        O2_delivery = O2_capacity * cardiac_output * 10  # mL O2/min (convert dL to L)
        
        # Tissue oxygen consumption (basal metabolic rate)
        O2_consumption = 250.0  # mL O2/min at rest
        
        # Tissue O2 saturation represents adequacy of oxygen delivery
        # Normal: delivery ~4x consumption → 95% saturation
        # Anemia/heart failure: delivery drops → saturation drops → EPO rises
        
        baseline_delivery = 1000.0  # Normal O2 delivery (mL/min)
        baseline_saturation = 95.0  # Normal tissue O2 saturation (%)
        
        # Saturation proportional to delivery relative to baseline
        tissue_O2_saturation = baseline_saturation * (O2_delivery / baseline_delivery)
        tissue_O2_saturation = max(0.0, min(100.0, tissue_O2_saturation))
        
        state.set_organism_state('tissue_oxygen_saturation', tissue_O2_saturation)