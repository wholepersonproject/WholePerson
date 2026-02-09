import numpy as np
from models.base import ProcessModel, TimeScale

class CalcitoninSecretion_0036161(ProcessModel): 
    inputs = {
        'blood_calcium': ('blood', 'calcium'),
        'blood_calcitonin': ('blood', 'calcitonin')

    }
    outputs = {
        'blood_calcitonin': ('blood', 'calcitonin')
    }
    
    parameters = {
        'calcium_threshold': {
            'default': 20, 
            'unit': 'pg/mL', 
            'description': "Threshold to increase secretion"
        },
        'calcium_activation_coeff': {
            'default': 2.0, 
            'unit': 'NA', 
        },
        'calcitonin_secretion_rate': {
            'default': 5.0,
            'unit': 'pg/mL/min',
            'range': (5.0, 100.0),
            'description': 'Baseline parafollicular cell calcitonin secretion'
        }
    }

    def __init__(self, calcitonin_secretion_rate = 5.0, calcium_threshold = 20, calcium_activation_coeff = 2.0):
        super().__init__("calcitonin_secretion", TimeScale.MINUTES)
        self.calcitonin_secretion_rate =  calcitonin_secretion_rate
        self.calcium_threshold = calcium_threshold
        self.calcium_activation_coeff = calcium_activation_coeff

    def step(self, state, dt):
        ca_level = state.get_signal('blood', 'calcium')
        calcitonin_level = state.get_signal('blood', 'calcitonin')

        parafollicular_agents = state.get_agents('parafollicular_cells')

        total_calcitonin_delta = 0

        for agent in parafollicular_agents:
            #Sense the blood calcium levels and release calcitonin appropriately
            if(ca_level > self.calcium_threshold):
                delta_ca = ca_level - self.calcium_threshold
                secretion_multiplier = delta_ca * self.calcium_activation_coeff + 1.0
            else: 
                # low levels of calcium already
                secretion_multiplier = 0.1
            #  
            calcitonin_from_cell = self.calcitonin_secretion_rate * secretion_multiplier * dt/60
            total_calcitonin_delta += calcitonin_from_cell


        new_calcitonin = calcitonin_level + total_calcitonin_delta

        state.set_signal('blood', 'calcitonin', new_calcitonin)

class ParathyroidHormoneSecretion_0035898(ProcessModel):
    inputs = {
        'blood_parathyroid': ('blood', 'parathyroid_hormone'),
        'blood_calcium': ('blood', 'calcium')

    }
    outputs = {
        'blood_parathyroid': ('blood', 'parathyroid_hormone')
    }
    
    parameters = {
        'calcium_threshold': {
            'default': 20, 
            'unit': 'pg/mL', 
            'description': "Threshold to decrease secretion"
        },
        'calcium_decrease_coeff': {
            'default': 2.0, 
            'unit': 'NA', 
        },
        'parathyroid_hormone_secretion_rate': {
            'default': 5.0,
            'unit': 'pg/mL/min',
            'range': (5.0, 100.0),
            'description': 'Baseline chief cell parathyroid hormone secretion'
        }
    }

    def __init__(self, parathyroid_hormone_secretion_rate = 5.0, calcium_threshold = 20, calcium_decrease_coeff = 2.0):
        super().__init__("parathyroid_secretion", TimeScale.MINUTES)
        self.parathyroid_hormone_secretion_rate =  parathyroid_hormone_secretion_rate
        self.calcium_threshold = calcium_threshold
        self.calcium_decrease_coeff = calcium_decrease_coeff

    def step(self, state, dt):
        ca_level = state.get_signal('blood', 'calcium')
        PTH_level = state.get_signal('blood', 'parathyroid_hormone')

        chief_cell_agents = state.get_agents('chief_cells')

        PTH_delta = 0

        for agent in chief_cell_agents:
            #Sense the blood calcium levels and release PTH appropriately
            # Hill-function based PTH secretion
            K = 9.5  # calcium at half-max secretion
            n = 4    # cooperativity
            PTH_from_cell = self.parathyroid_hormone_secretion_rate * (K**n / (K**n + ca_level**n)) * dt

            PTH_delta += PTH_from_cell

        new_PTH = PTH_level + PTH_delta

        state.set_signal('blood', 'parathyroid_hormone', new_PTH)

class DCTCalciumReabsorption_0035898(ProcessModel):
    """
    PTH-regulated active calcium reabsorption in distal convoluted tubule
    
    Mechanism: PTH upregulates TRPV5 (apical entry), calcitriol upregulates
    calbindin-D28k (intracellular shuttle — modeled as internal state).
    Transport is rate-limited (Michaelis-Menten), not fractional extraction.
    Timescale: Minutes (PTH effect), hours (calbindin turnover)
    Location: Kidney DCT
    
    Equations:
    
        Internal ODE (calbindin, slow):
            calbindin_target = min(calcitriol / 50, 1.5)
            d(calbindin)/dt = (calbindin_target - calbindin) / tau
            tau = 6 hours
    
        Transport capacity (algebraic):
            TRPV5_activation = PTH / (K_pth + PTH)
            Vmax = Vmax_basal + (Vmax_max - Vmax_basal) × TRPV5_activation × calbindin
    
        Active transport (Michaelis-Menten):
            transport_rate = Vmax × Ca_lumen / (Km + Ca_lumen)
            reabsorbed = transport_rate × dt
    """
    
    inputs = {
        'pth':              ('blood', 'parathyroid_hormone'),
        'calcitriol':       ('blood', 'calcitriol'),
        'filtered_calcium': ('kidney', 'filtered_calcium'),
    }
    outputs = {
        'blood_calcium':    ('blood', 'calcium'),
        'filtered_calcium': ('kidney', 'filtered_calcium'),
    }
    
    parameters = {
        'Vmax_basal': {
            'default': 0.40,
            'unit': 'mg/min',
            'range': (0.2, 0.6),
            'description': 'Baseline transport capacity without PTH stimulation'
        },
        'Vmax_max': {
            'default': 1.1,
            'unit': 'mg/min',
            'range': (0.8, 1.5),
            'description': 'Maximum transport capacity at full TRPV5 activation and calbindin'
        },
        'Km': {
            'default': 0.3,
            'unit': 'mg',
            'range': (0.1, 0.8),
            'description': 'TRPV5 affinity for luminal calcium — intrinsic to channel, not regulated'
        },
        'K_pth': {
            'default': 20.0,
            'unit': 'pg/mL',
            'range': (10.0, 40.0),
            'description': 'TRPV5 receptor sensitivity — PTH for half-maximal channel activation'
        },
    }
    
    def __init__(self, Vmax_basal=0.40, Vmax_max=1.1, Km=0.3, K_pth=20.0):
        super().__init__("dct_calcium_reabsorption", TimeScale.MINUTES)
        self.Vmax_basal = Vmax_basal
        self.Vmax_max = Vmax_max
        self.Km = Km
        self.K_pth = K_pth
        
        # Internal state — not shared, never read by other processes
        self.calbindin_level = 1.0
    
    def step(self, state, dt):
        pth = state.get_signal('blood', 'parathyroid_hormone')
        calcitriol = state.get_signal('blood', 'calcitriol')
        filtered_ca = state.get_signal('kidney', 'filtered_calcium')
        
        # --- Internal ODE: calcitriol slowly regulates calbindin ---
        calcitriol_normal = 50.0  # pg/mL
        calbindin_target = min(calcitriol / calcitriol_normal, 1.5)
        tau_calbindin = 6.0 * 3600.0  # ~6 hour turnover
        self.calbindin_level += (calbindin_target - self.calbindin_level) * (dt / tau_calbindin)
        
        # --- Transport capacity: PTH opens channels, calbindin sets throughput ---
        trpv5_activation = pth / (self.K_pth + pth)
        Vmax = self.Vmax_basal + (self.Vmax_max - self.Vmax_basal) * trpv5_activation * self.calbindin_level
        
        # --- Michaelis-Menten active transport ---
        transport_rate = Vmax * filtered_ca / (self.Km + filtered_ca)  # mg/min
        reabsorbed = min(transport_rate * (dt / 60.0), filtered_ca)    # can't reabsorb more than available
        
        # Return calcium to blood
        state.update_signal('blood', 'calcium', reabsorbed / 50.0)  # mg → mg/dL in 5L
        state.update_signal('kidney', 'filtered_calcium', -reabsorbed)



class ProximalTubuleHydroxylase(ProcessModel):
    """
    1α-hydroxylase reaction in kidney proximal tubule
    
    Mechanism: Converts calcifediol (25-OH-D, liver output) to calcitriol
    (1,25-(OH)₂D₃, the active hormone). PTH induces enzyme expression.
    FGF23 binds FGFR1/Klotho on the basolateral membrane and both
    suppresses 1α-hydroxylase transcription and induces 24-hydroxylase
    (the competing degradation enzyme). High blood calcium directly
    suppresses 1α-hydroxylase independent of PTH.
    Timescale: Hours (enzyme induction)
    Location: Kidney proximal tubule epithelial cells
    
    Equations:
        substrate = calcifediol / (K_calcifediol + calcifediol)
        pth_induction = PTH / (K_pth + PTH)
        fgf23_suppression = 1 / (1 + FGF23 / K_fgf23)
        ca_suppression = clamp(1.0 - 0.3 × (Ca - 9.5) / 9.5, 0.2, 1.3)
        rate = Vmax × substrate × pth_induction × fgf23_suppression × ca_suppression
        d(calcitriol)/dt += rate
        d(calcifediol)/dt -= rate × conversion_ratio
    """
    
    inputs = {
        'calcifediol': ('blood', 'calcifediol'),
        'pth':         ('blood', 'pth'),
        'fgf23':       ('blood', 'fgf23'),
        'calcium':     ('blood', 'calcium'),
    }
    outputs = {
        'calcitriol':  ('blood', 'calcitriol'),
        'calcifediol': ('blood', 'calcifediol'),
    }
    
    parameters = {
        'Vmax': {
            'default': 16.0,
            'unit': 'pg/mL/hr',
            'range': (8.0, 30.0),
            'description': 'Maximum 1α-hydroxylase reaction rate'
        },
        'K_calcifediol': {
            'default': 15.0,
            'unit': 'ng/mL',
            'range': (8.0, 25.0),
            'description': '1α-hydroxylase affinity for calcifediol substrate'
        },
        'K_pth': {
            'default': 40.0,
            'unit': 'pg/mL',
            'range': (20.0, 80.0),
            'description': '1α-hydroxylase induction sensitivity to PTH'
        },
        'K_fgf23': {
            'default': 40.0,
            'unit': 'RU/mL',
            'range': (20.0, 80.0),
            'description': 'FGFR1/Klotho sensitivity — FGF23 for half-maximal 1α-hydroxylase suppression'
        },
        'conversion_ratio': {
            'default': 0.001,
            'unit': 'ng/mL per pg/mL',
            'range': (0.0005, 0.005),
            'description': 'Calcifediol consumed per calcitriol produced (unit scaling: ng/mL vs pg/mL)'
        },
    }
    
    def __init__(self, Vmax=16.0, K_calcifediol=15.0, K_pth=40.0, K_fgf23=40.0, conversion_ratio=0.001):
        super().__init__("proximal_tubule_hydroxylase", TimeScale.HOURS)
        self.Vmax = Vmax
        self.K_calcifediol = K_calcifediol
        self.K_pth = K_pth
        self.K_fgf23 = K_fgf23
        self.conversion_ratio = conversion_ratio
    
    def step(self, state, dt):
        calcifediol = state.get_signal('blood', 'calcifediol')
        pth = state.get_signal('blood', 'pth')
        fgf23 = state.get_signal('blood', 'fgf23')
        ca = state.get_signal('blood', 'calcium')
        
        # Substrate availability (Michaelis-Menten on calcifediol)
        substrate = calcifediol / (self.K_calcifediol + calcifediol)
        
        # PTH induces 1α-hydroxylase expression
        pth_induction = pth / (self.K_pth + pth)
        
        # FGF23 suppresses 1α-hydroxylase and induces competing 24-hydroxylase
        fgf23_suppression = 1.0 / (1.0 + fgf23 / self.K_fgf23)
        
        # High calcium directly suppresses 1α-hydroxylase
        ca_suppression = 1.0 - 0.3 * ((ca - 9.5) / 9.5)
        ca_suppression = np.clip(ca_suppression, 0.2, 1.3)
        
        # Reaction rate
        rate = self.Vmax * substrate * pth_induction * fgf23_suppression * ca_suppression
        
        # Calcitriol produced, calcifediol consumed
        state.update_signal('blood', 'calcitriol', rate * (dt / 3600.0))
        state.update_signal('blood', 'calcifediol', -rate * self.conversion_ratio * (dt / 3600.0))