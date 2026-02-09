import numpy as np
from models.base import ProcessModel, TimeScale

class InsulinSecretion(ProcessModel):
    """
    Beta-cell insulin secretion in response to glucose
    
    Mechanism: Sigmoidal (Hill) dose-response
    Timescale: Minutes
    Location: Pancreatic beta cells
    """
    
    inputs = {
        'glucose': ('blood', 'glucose')  # ← CHANGED: Read from blood
    }
    outputs = {
        'blood_insulin': ('blood', 'insulin')
    }
    
    parameters = {
        'glucose_sensitivity': {
            'default': 1.0,
            'unit': 'dimensionless',
            'range': (0.3, 2.0),
            'description': 'Beta-cell responsiveness; 1.0=normal, <0.5=insulin resistant, >1.5=highly sensitive'
        },
        'basal_secretion': {
            'default': 5.0,
            'unit': 'µU/mL/min',
            'range': (2.0, 10.0),
            'description': 'Fasting insulin secretion rate'
        },
        'max_secretion': {
            'default': 150.0,
            'unit': 'µU/mL/min',
            'range': (50.0, 250.0),
            'description': 'Maximum glucose-stimulated secretion'
        }
    }
    
    def __init__(self, glucose_sensitivity=1.0, basal_secretion=5.0, max_secretion=150.0):
        super().__init__("insulin_secretion", TimeScale.MINUTES)
        self.glucose_sensitivity = glucose_sensitivity
        self.basal_secretion = basal_secretion
        self.max_secretion = max_secretion
    
    def step(self, state, dt):
        glucose = state.get_signal('blood', 'glucose')  # ← CHANGED: From blood
        
        K = 90.0  # mg/dL - half-maximal glucose
        n = 1.7   # Hill coefficient
        
        stimulated = (self.max_secretion - self.basal_secretion) * \
                     (glucose**n) / (K**n + glucose**n)
        secretion_rate = self.basal_secretion + stimulated * self.glucose_sensitivity
        
        state.update_signal('blood', 'insulin', secretion_rate * (dt/60.0))


class GlucagonSecretion(ProcessModel):
    """
    Alpha-cell glucagon secretion
    
    Mechanism: Inverse glucose response with insulin inhibition
    Timescale: Minutes
    """
    
    inputs = {
        'glucose': ('blood', 'glucose'),
        'insulin': ('blood', 'insulin')
    }
    outputs = {
        'blood_glucagon': ('blood', 'glucagon')
    }
    
    parameters = {
        'basal_secretion': {
            'default': 60.0,
            'unit': 'pg/mL/min',
            'range': (40.0, 100.0),
            'description': 'Baseline alpha-cell secretion'
        }
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


class GlucoseUptake(ProcessModel):
    """
    Insulin-mediated glucose uptake by tissues
    
    Mechanism: Basal + insulin-stimulated GLUT4 transport
    Timescale: Minutes
    """
    
    parameters = {
        'basal_rate': {
            'default': 0.1,
            'unit': 'mg/dL/min',
            'range': (0.005, 0.5),
            'description': 'Insulin-independent glucose uptake'
        },
        'insulin_sensitivity': {
            'default': 1.0,
            'unit': 'dimensionless',
            'range': (0.2, 2.5),
            'description': 'Tissue insulin sensitivity; 1.0=normal, <0.5=resistant, >1.5=athlete'
        }
    }
    
    def __init__(self, target_entity='muscle_tissue', basal_rate=0.1, insulin_sensitivity=1.0):
        super().__init__(f"glucose_uptake_{target_entity}", TimeScale.MINUTES)
        self.target_entity = target_entity
        self.basal_rate = basal_rate
        self.insulin_sensitivity = insulin_sensitivity
        
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
    """
    Glycogen synthesis in liver and muscle
    
    Mechanism: Insulin-activated glycogen synthase
    Timescale: Hours
    """
    
    inputs = {
        'fed_status': ('fed_status', None, 'organism'),
        'liver_glucose': ('liver', 'glucose'),
        'blood_insulin': ('blood', 'insulin'),  # ← CHANGED: From blood
        'muscle_glucose': ('muscle_tissue', 'glucose'),
    }
    outputs = {
        'liver_glucose': ('liver', 'glucose'),
        'liver_glycogen': ('liver', 'glycogen'),
        'muscle_glucose': ('muscle_tissue', 'glucose'),
        'muscle_glycogen': ('muscle_tissue', 'glycogen')
    }
    
    parameters = {}
    
    def __init__(self):
        super().__init__("glycogen_synthesis", TimeScale.HOURS)
    
    def step(self, state, dt):
        fed_status = state.get_organism_state('fed_status', 'fasted')
        
        if fed_status == 'fed':
            liver_glucose = state.get_signal('liver', 'glucose')
            liver_insulin = state.get_signal('blood', 'insulin')  # ← CHANGED: From blood
            
            if liver_insulin > 8.0:
                synthesis = 0.05 * liver_glucose * (liver_insulin / 10.0)
                amount = synthesis * (dt / 3600.0)
                state.update_signal('liver', 'glucose', -amount)
                state.update_signal('liver', 'glycogen', amount)
            
            muscle_glucose = state.get_signal('muscle_tissue', 'glucose')
            muscle_insulin = state.get_signal('blood', 'insulin')  # ← CHANGED: From blood
            
            if muscle_insulin > 8.0:
                synthesis = 0.03 * muscle_glucose * (muscle_insulin / 10.0)
                amount = synthesis * (dt / 3600.0)
                state.update_signal('muscle_tissue', 'glucose', -amount)
                state.update_signal('muscle_tissue', 'glycogen', amount)


class GlycogenBreakdown(ProcessModel):
    """
    Glycogen breakdown (glycogenolysis) in liver
    
    Mechanism: Glucagon-activated glycogen phosphorylase
    Timescale: Minutes
    """
    
    inputs = {
        'fed_status': ('fed_status', None, 'organism'),
        'glucagon': ('blood', 'glucagon'),
        'glycogen': ('liver', 'glycogen')
    }
    outputs = {
        'liver_glycogen': ('liver', 'glycogen'),
        'blood_glucose': ('blood', 'glucose')
    }
    
    parameters = {}
    
    def __init__(self):
        super().__init__("glycogen_breakdown", TimeScale.MINUTES)  # ← CHANGED from HOURS
    
    def step(self, state, dt):
        fed_status = state.get_organism_state('fed_status', 'fasted')
        glucagon = state.get_signal('blood', 'glucagon')
        
        if fed_status == 'fasted' and glucagon > 70.0:
            liver_glycogen = state.get_signal('liver', 'glycogen')
            
            # Breakdown rate in g/min (not g/hour!)
            # At glucagon=80, breakdown = 0.5 g/min
            breakdown_rate_per_min = 0.5 * (glucagon / 80.0)
            
            # Amount in this timestep (dt is in seconds)
            amount_grams = breakdown_rate_per_min * (dt / 60.0)
            
            # Don't break down more than available
            amount_grams = min(amount_grams, liver_glycogen)
            
            # Remove from glycogen store
            state.update_signal('liver', 'glycogen', -amount_grams)
            
            # Convert grams of glycogen to mg/dL blood glucose
            # 1g glycogen → ~1g glucose (molecular weight similar)
            # 1g glucose = 1000 mg
            # Blood volume = 5 L = 50 dL
            # So 1g glucose → 1000mg / 50dL = 20 mg/dL concentration increase
            glucose_release_mg_per_dL = amount_grams * 20.0
            
            # Add to blood glucose
            state.update_signal('blood', 'glucose', glucose_release_mg_per_dL)


class HepaticGlucoseProduction(ProcessModel):
    """
    Hepatic glucose production (gluconeogenesis)
    
    Mechanism: Glucagon-stimulated, insulin-suppressed
    Timescale: Hours
    """
    
    inputs = {
        'glucagon': ('blood', 'glucagon'),
        'insulin': ('blood', 'insulin'),
        'fed_status': ('fed_status', None, 'organism')
    }
    outputs = {
        'blood_glucose': ('blood', 'glucose')
    }
    
    parameters = {
        'production_rate': {
            'default': 2.0,
            'unit': 'mg/kg/min',
            'range': (1.5, 3.0),
            'description': 'Basal hepatic glucose output'
        }
    }
    
    def __init__(self, production_rate=2.0):
        super().__init__("hepatic_glucose_production", TimeScale.MINUTES)  # ← CHANGE to MINUTES
        self.production_rate = production_rate
    
    def step(self, state, dt):
        glucagon = state.get_signal('blood', 'glucagon')
        insulin = state.get_signal('blood', 'insulin')
        fed_status = state.get_organism_state('fed_status', 'fasted')
        
        glucagon_factor = glucagon / 60.0
        insulin_factor = 1.0 / (1.0 + insulin / 10.0)
        fasting_boost = 1.5 if fed_status == 'fasted' else 0.5
        
        # Calculate production in mg/min for whole body
        body_weight_kg = 70.0
        blood_volume_dL = 50.0  # 5 L
        
        production_mg_per_min = self.production_rate * glucagon_factor * insulin_factor * fasting_boost * body_weight_kg
        production_concentration_per_min = production_mg_per_min / blood_volume_dL  # mg/dL/min
        
        # dt is in seconds for MINUTES timescale
        amount = production_concentration_per_min * (dt / 60.0)
        
        state.update_signal('blood', 'glucose', amount)


class HormoneDegradation(ProcessModel):
    """
    Hormone clearance and degradation
    
    Mechanism: First-order clearance (hepatic/renal)
    Timescale: Minutes
    """
    
    inputs = {
        'blood_insulin': ('blood', 'insulin'),
        'blood_glucagon': ('blood', 'glucagon'),
        'blood_erythropoietin': ('blood', 'erythropoietin'),
        'blood_calcitonin': ('blood', 'calcitonin'),
        'blood_parathyroid': ('blood', 'parathyroid_hormone')

    }
    outputs = {
        'blood_insulin': ('blood', 'insulin'),
        'blood_glucagon': ('blood', 'glucagon'),
        'blood_erythropoietin': ('blood', 'erythropoietin'), 
        'blood_calcitonin': ('blood', 'calcitonin'),
        'blood_parathyroid': ('blood', 'parathyroid_hormone')

    }
    
    parameters = {
        'insulin_half_life': {
            'default': 5.0,
            'unit': 'minutes',
            'range': (3.0, 8.0),
            'description': 'Insulin plasma half-life'
        },
        'glucagon_half_life': {
            'default': 6.0,
            'unit': 'minutes',
            'range': (4.0, 10.0),
            'description': 'Glucagon plasma half-life'
        },
        'erythropoietin_half_life': {
            'default': 300.0,
            'unit': 'minutes',
            'range': (4.0, 10.0),
            'description': 'erythropoietin plasma half-life'
        },
        'calcitonin_half_life': {
            'default': 5.0,
            'unit': 'minutes',
            'range': (4.0, 10.0),
            'description': 'calcitonin plasma half-life'
        },
        'parathyroid_hormone_half_life': {
            'default': 5.0,
            'unit': 'minutes',
            'range': (4.0, 10.0),
            'description': 'parathyroid hormone plasma half-life'
        }
        
        
        
    }
    
    def __init__(self, insulin_half_life=5.0, glucagon_half_life=6.0, erythropoietin_half_life = 300.0, calcitonin_half_life = 5.0, parathyroid_hormone_half_life = 5.0):
        super().__init__("hormone_degradation", TimeScale.MINUTES)
        self.insulin_half_life = insulin_half_life
        self.glucagon_half_life = glucagon_half_life
        self.erythropoietin_half_life = erythropoietin_half_life
        self.calcitonin_half_life = calcitonin_half_life
        self.parathyroid_hormone_half_life = parathyroid_hormone_half_life
    
    def step(self, state, dt):
        insulin = state.get_signal('blood', 'insulin')
        glucagon = state.get_signal('blood', 'glucagon')
        erythropoietin = state.get_signal('blood', 'erythropoietin')
        calcitonin = state.get_signal('blood', 'calcitonin')
        PTH = state.get_signal('blood', 'parathyroid_hormone')
        
        insulin_decay = insulin * (1 - np.exp(-np.log(2) * dt / (self.insulin_half_life * 60)))
        glucagon_decay = glucagon * (1 - np.exp(-np.log(2) * dt / (self.glucagon_half_life * 60)))
        erythropoietin_decay = erythropoietin * (1 - np.exp(-np.log(2) * dt / (self.erythropoietin_half_life * 60)))
        calcitonin_decay = calcitonin * (1 - np.exp(-np.log(2) * dt / (self.calcitonin_half_life * 60)))
        pth_decay = PTH * (1 - np.exp(-np.log(2) * dt / (self.parathyroid_hormone_half_life * 60)))

        
        state.update_signal('blood', 'insulin', -insulin_decay)
        state.update_signal('blood', 'glucagon', -glucagon_decay)
        state.update_signal('blood', 'erythropoietin', -erythropoietin_decay)
        state.update_signal('blood', 'calcitonin', -calcitonin_decay)
        state.update_signal('blood', 'parathyroid_hormone', -pth_decay)


class f_cell_polypeptide_0036322(ProcessModel):
    """
    Pancreatic F-cell polypeptide (PP) secretion
    
    Mechanism: Agent-based F-cell dynamics
    Timescale: Minutes
    """
    
    inputs = {
        'fed_status': ('fed_status', None, 'organism'),
        'blood_pp': ('blood', 'pancreatic_polypeptide')
    }
    outputs = {
        'blood_pp': ('blood', 'pancreatic_polypeptide')
    }
    
    parameters = {}
    
    def __init__(self):
        super().__init__("f_cell_dynamics", TimeScale.MINUTES)
    
    def step(self, state, dt):
        fed_status = state.get_organism_state('fed_status', 'fasted')
        pp_level = state.get_signal('blood', 'pancreatic_polypeptide')
        
        agents = state.get_agents('f_cells')
        
        total_secretion = 0.0
        secretion_capacity = 0.5
        
        for agent in agents:
            if 'state' not in agent:
                agent['state'] = {
                    'secretion_capacity': secretion_capacity,
                    'max_capacity': secretion_capacity
                }
            
            if fed_status == 'fed':
                secretion = agent['state']['secretion_capacity'] * 0.1 * (dt / 60)
                total_secretion += secretion
                agent['state']['secretion_capacity'] *= 0.9999
            else:
                if agent['state']['secretion_capacity'] < agent['state'].get('max_capacity', secretion_capacity):
                    agent['state']['secretion_capacity'] *= 1.0001
        
        clearance = pp_level * 0.12 * (dt / 60)
        net_change = total_secretion - clearance
        
        new_pp = max(10, pp_level + net_change)
        state.set_signal('blood', 'pancreatic_polypeptide', new_pp)


class ErythropoietinProduction(ProcessModel):
    """
    EPO production by kidneys in response to tissue hypoxia
    """
    
    parameters = {
        'basal_epo': {
            'default': 10.0,
            'unit': 'mU/mL',
            'range': (5.0, 20.0),
            'description': 'Baseline EPO in normoxia'
        },
        'max_epo': {
            'default': 200.0,
            'unit': 'mU/mL',
            'range': (100.0, 1000.0),
            'description': 'Maximum EPO in severe hypoxia'
        }
    }
    
    def __init__(self, basal_epo=10.0, max_epo=200.0):
        super().__init__("epo_production", TimeScale.HOURS)
        self.basal_epo = basal_epo
        self.max_epo = max_epo
    
    def step(self, state, dt):
        tissue_O2 = state.get_organism_state('tissue_oxygen_saturation', 95.0)
        epo_capacity = state.get_signal('kidney', 'epo_production_capacity')
        if epo_capacity is None:
            epo_capacity = 1.0
        kidney_mass = state.get_signal('kidney', 'functional_mass')
        if kidney_mass is None:
            kidney_mass = 1.0
        
        # Target O2 should match baseline tissue O2
        target_O2 = 97.0  # ← CHANGE from 95.0 to match your baseline
        
        # Hypoxia factor: 0 when O2 >= target, increases as O2 drops
        hypoxia_factor = max(0.0, (target_O2 - tissue_O2) / target_O2)
        
        # Hill function for sigmoidal response
        hill_n = 2.5
        K_half = 0.15  # ← INCREASE from 0.1 (less sensitive)
        
        epo_production_rate = self.basal_epo + \
            (self.max_epo - self.basal_epo) * \
            (hypoxia_factor**hill_n) / (K_half**hill_n + hypoxia_factor**hill_n)
        
        # Scale by kidney function
        epo_production_rate *= epo_capacity * kidney_mass
        
        # Calculate amount (mU/mL change)
        # This is production rate per hour
        epo_amount = epo_production_rate * (dt / 3600.0)
        
        # Get current EPO for negative feedback
        current_epo = state.get_signal('blood', 'erythropoietin')
        if current_epo is None:
            current_epo = self.basal_epo
        
        # Add negative feedback: reduce production if EPO is high
        if current_epo > self.basal_epo * 2:
            feedback_factor = self.basal_epo * 2 / current_epo
            epo_amount *= feedback_factor
        
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
        epo = state.get_signal('blood', 'erythropoietin')
        if epo is None:
            epo = 10.0
        
        marrow_capacity = state.get_signal('bone_marrow', 'erythropoiesis_capacity')
        if marrow_capacity is None:
            marrow_capacity = 1.0
        
        # Get current RBC for negative feedback
        current_rbc = state.get_signal('blood', 'rbc_count')
        if current_rbc is None:
            current_rbc = 5.0e6
        
        # Add negative feedback: stop production if RBC too high
        target_rbc = 5.5e6  # Normal upper limit
        if current_rbc > target_rbc:
            rbc_inhibition = target_rbc / current_rbc
        else:
            rbc_inhibition = 1.0
        
        epo_baseline = 10.0
        K_epo = 20.0
        hill_n = 2.0
        epo_factor = 1.0 + 3.0 * (epo**hill_n) / (K_epo**hill_n + epo**hill_n)
        
        rbc_production_rate = self.basal_production * epo_factor * marrow_capacity * rbc_inhibition
        
        blood_volume_liters = 5.0
        new_rbcs = rbc_production_rate * (dt / 86400.0)
        rbc_per_uL_increase = new_rbcs / (blood_volume_liters * 1e6)
        
        state.update_signal('blood', 'rbc_count', rbc_per_uL_increase)
        
        hb_increase = (new_rbcs * 30e-12) / (blood_volume_liters * 10)
        state.update_signal('blood', 'hemoglobin', hb_increase)


class RBCTurnover(ProcessModel):
    """
    RBC removal by spleen (120-day lifespan)
    
    Mechanism: Senescent RBC removal by reticuloendothelial system
    Timescale: Days
    """
    
    inputs = {
        'rbc_count': ('blood', 'rbc_count'),
        'hemoglobin': ('blood', 'hemoglobin')
    }
    outputs = {
        'blood_rbc_count': ('blood', 'rbc_count'),
        'blood_hemoglobin': ('blood', 'hemoglobin')
    }
    
    parameters = {
        'rbc_lifespan_days': {
            'default': 120.0,
            'unit': 'days',
            'range': (90.0, 140.0),
            'description': 'RBC lifespan; reduced in hemolytic anemia'
        }
    }
    
    def __init__(self, rbc_lifespan_days=120.0):
        super().__init__("rbc_turnover", TimeScale.DAYS)
        self.rbc_lifespan_days = rbc_lifespan_days
    
    def step(self, state, dt):
        rbc_count = state.get_signal('blood', 'rbc_count')
        hemoglobin = state.get_signal('blood', 'hemoglobin')
        
        if rbc_count is None or rbc_count == 0:
            rbc_count = 5.0e6
        if hemoglobin is None or hemoglobin == 0:
            hemoglobin = 15.0
        
        removal_rate = 1.0 / self.rbc_lifespan_days
        
        rbc_removed = rbc_count * removal_rate * (dt / 86400.0)
        hb_removed = hemoglobin * removal_rate * (dt / 86400.0)
        
        state.update_signal('blood', 'rbc_count', -rbc_removed)
        state.update_signal('blood', 'hemoglobin', -hb_removed)


class OxygenDelivery(ProcessModel):
    """
    Oxygen transport from lungs to tissues via hemoglobin
    
    Mechanism: Hb-O2 binding and dissociation
    Timescale: Seconds (circulation time)
    """
    
    inputs = {
        'hemoglobin': ('blood', 'hemoglobin'),
        'rbc_count': ('blood', 'rbc_count'),
        'cardiac_output': ('cardiac_output', None, 'organism')
    }
    outputs = {
        'tissue_O2_saturation': ('tissue_oxygen_saturation', None, 'organism')
    }
    
    parameters = {}
    
    def __init__(self):
        super().__init__("oxygen_delivery", TimeScale.SECONDS)
    
    def step(self, state, dt):
        hemoglobin = state.get_signal('blood', 'hemoglobin')
        if hemoglobin is None or hemoglobin == 0:
            hemoglobin = 15.0
        
        cardiac_output = state.get_organism_state('cardiac_output', 5.0)
        
        O2_capacity = hemoglobin * 1.34
        O2_delivery = O2_capacity * cardiac_output * 10
        O2_consumption = 250.0
        
        baseline_delivery = 1000.0
        baseline_saturation = 95.0
        
        tissue_O2_saturation = baseline_saturation * (O2_delivery / baseline_delivery)
        tissue_O2_saturation = max(0.0, min(100.0, tissue_O2_saturation))
        
        state.set_organism_state('tissue_oxygen_saturation', tissue_O2_saturation)

class TissueGlucoseConsumption(ProcessModel):
    """
    Glucose oxidation by tissues for ATP production
    
    Mechanism: Glycolysis + TCA cycle
    Timescale: Minutes
    """
    
    parameters = {
        'target_entity': {
            'default': 'muscle_tissue'
        },
        'consumption_rate': {
            'default': 0.05,
            'unit': 'mg/dL/min',
            'description': 'Basal glucose consumption rate'
        }
    }
    
    def __init__(self, target_entity='muscle_tissue', consumption_rate=0.05):
        super().__init__(f"glucose_consumption_{target_entity}", TimeScale.MINUTES)
        self.target_entity = target_entity
        self.consumption_rate = consumption_rate
        
        self.inputs = {
            'tissue_glucose': (target_entity, 'glucose')
        }
        self.outputs = {
            'tissue_glucose': (target_entity, 'glucose')
        }
    
    def step(self, state, dt):
        tissue_glucose = state.get_signal(self.target_entity, 'glucose')
        
        # Consume glucose proportional to availability
        consumption = self.consumption_rate * (tissue_glucose / 90.0)
        amount = consumption * (dt / 60.0)
        
        # Don't consume more than available
        amount = min(amount, tissue_glucose)
        
        state.update_signal(self.target_entity, 'glucose', -amount)