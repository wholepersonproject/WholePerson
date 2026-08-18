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
            PTH_from_cell = self.parathyroid_hormone_secretion_rate * (K**n / (K**n + ca_level**n)) * (dt/60)

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
        'pth': ('blood', 'parathyroid_hormone'),
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
        pth = state.get_signal('blood', 'parathyroid_hormone')
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


class SclerostinRegulation_0046850(ProcessModel):
    """
    Osteocyte mechanosensing and sclerostin regulation on spatial bone grid
    
    Mechanism: Osteocytes are the primary mechanosensors in bone, embedded
    in the mineralized matrix. Mechanical strain drives canalicular fluid
    flow past osteocyte processes, activating mechanotransduction (Piezo1,
    integrins, primary cilia). Under sufficient strain, osteocytes suppress
    sclerostin (SOST) transcription. Sclerostin is a Wnt antagonist —
    its reduction de-represses the Wnt/β-catenin pathway in nearby
    osteoblasts, enabling bone formation.
    
    Sclerostin is secreted locally and diffuses through the bone matrix
    within paracrine range (~2–3 mm). It decays with a ~6 hour half-life.
    
    The strain field is computed from organism exercise_intensity, mapped
    to a spatial gradient via Frost's mechanostat framework: periosteal
    (near-surface) voxels see more strain than endosteal (interior).
    
    Timescale: Minutes (mechanotransduction is fast; protein turnover is slow
               but the secretion decision is rapid)
    Location: Bone tissue (osteocyte lacunar-canalicular network)
    
    CSV rows covered:
        - osteocyte does sclerostin secretion          [2.5, 4] ng/mL
        - mechanical loading decreases sclerostin      (systemic → paracrine)
        - sclerostin decreases bone synthesis           (via diffusion to osteoblasts)
    
    Equations:
        Strain field (algebraic, recomputed each step):
            depth[i,j,k] = min distance to grid boundary
            surface_strain = base_strain + (peak_strain - base_strain) × exercise
            strain[i,j,k] = surface_strain × (1 - attenuation × depth / max_depth)
    
        Per osteocyte at grid position (i,j,k):
            local_strain = strain[i,j,k]
            suppression = K_strain^n / (K_strain^n + local_strain^n)
            secretion = base_rate × agent.mechanosensitivity × suppression × dt
            sclerostin_field[i,j,k] += secretion
    
        Diffusion (3D explicit finite difference, Neumann BC):
            ∇²S = Laplacian stencil with no-flux padding
            S += D × ∇²S × dt
    
        Decay:
            S *= exp(-decay_rate × dt)
    """
    
    inputs = {
        'exercise':          ('organism', 'exercise_intensity'),
        'sclerostin_field':  ('bone', 'sclerostin'),
    }
    outputs = {
        'sclerostin_field':  ('bone', 'sclerostin'),
    }
    
    parameters = {
        'base_sclerostin_rate': {
            'default': 0.05,
            'unit': 'ng/mL/min/agent',
            'range': (0.01, 0.1),
            'description': 'Sclerostin secretion rate per osteocyte at zero strain'
        },
        'K_strain': {
            'default': 1500.0,
            'unit': 'microstrain',
            'range': (800.0, 2500.0),
            'description': 'Frost mechanostat set-point — strain for half-max secretion suppression'
        },
        'strain_cooperativity': {
            'default': 3,
            'unit': 'dimensionless',
            'range': (2, 5),
            'description': 'Hill coefficient for strain-response sigmoidal sharpness'
        },
        'diffusion_coeff': {
            'default': 0.001,
            'unit': 'mm²/s',
            'range': (0.0005, 0.005),
            'description': 'Sclerostin diffusion in bone matrix — gives ~3 mm paracrine range per hour'
        },
        'decay_rate': {
            'default': 3.2e-5,
            'unit': '1/s',
            'range': (1e-5, 1e-4),
            'description': 'Sclerostin degradation rate (~6 hour half-life, ln2/6h)'
        },
        'base_strain': {
            'default': 800.0,
            'unit': 'microstrain',
            'range': (400.0, 1200.0),
            'description': 'Background periosteal strain from sedentary daily activities'
        },
        'peak_exercise_strain': {
            'default': 2500.0,
            'unit': 'microstrain',
            'range': (1500.0, 4000.0),
            'description': 'Peak periosteal strain at maximal exercise_intensity'
        },
        'strain_attenuation': {
            'default': 0.7,
            'unit': 'dimensionless',
            'range': (0.5, 0.9),
            'description': 'Fraction of surface strain lost at deepest interior point'
        },
    }
    
    def __init__(self, base_sclerostin_rate=0.05, K_strain=1500.0,
                 strain_cooperativity=3, diffusion_coeff=0.001,
                 decay_rate=3.2e-5, base_strain=800.0,
                 peak_exercise_strain=2500.0, strain_attenuation=0.7):
        super().__init__("sclerostin_regulation", TimeScale.MINUTES)
        self.base_sclerostin_rate = base_sclerostin_rate
        self.K_strain = K_strain
        self.strain_cooperativity = strain_cooperativity
        self.diffusion_coeff = diffusion_coeff
        self.decay_rate = decay_rate
        self.base_strain = base_strain
        self.peak_exercise_strain = peak_exercise_strain
        self.strain_attenuation = strain_attenuation
        
        # Cached depth map — computed lazily on first step
        self._depth_map = None
        self._max_depth = None
    
    def _ensure_depth_map(self, shape):
        """Precompute normalized distance-from-boundary for the grid.
        Periosteal = 0 (surface), endosteal/interior = 1 (deepest).
        """
        if self._depth_map is not None:
            return
        nx, ny, nz = shape
        dx = np.minimum(np.arange(nx), nx - 1 - np.arange(nx))
        dy = np.minimum(np.arange(ny), ny - 1 - np.arange(ny))
        dz = np.minimum(np.arange(nz), nz - 1 - np.arange(nz))
        # Min distance to any face at each voxel
        dist = np.minimum(dx[:, None, None],
                          np.minimum(dy[None, :, None], dz[None, None, :]))
        self._max_depth = max(min(nx, ny, nz) // 2, 1)
        self._depth_map = dist.astype(float) / self._max_depth
    
    def _compute_strain_field(self, shape, exercise_intensity):
        """Map exercise_intensity to a spatial strain field (microstrain).
        
        Surface voxels receive full strain; interior attenuated by depth.
        This captures the key Wolff's law feature: cortical (periosteal)
        surfaces see high strain, cancellous interior sees less.
        """
        self._ensure_depth_map(shape)
        surface_strain = (self.base_strain
                          + (self.peak_exercise_strain - self.base_strain)
                          * exercise_intensity)
        strain_field = surface_strain * (1.0 - self.strain_attenuation * self._depth_map)
        return strain_field
    
    def _diffuse(self, field, dt, dx=1.0):
        """3D isotropic diffusion with no-flux (Neumann) boundary conditions.
        
        Explicit finite-difference. Stable for D×dt/dx² < 1/6.
        With D=0.001, dt=60s, dx=1mm: coefficient = 0.06, well within.
        """
        padded = np.pad(field, 1, mode='edge')
        laplacian = (padded[2:, 1:-1, 1:-1] + padded[:-2, 1:-1, 1:-1]
                     + padded[1:-1, 2:, 1:-1] + padded[1:-1, :-2, 1:-1]
                     + padded[1:-1, 1:-1, 2:] + padded[1:-1, 1:-1, :-2]
                     - 6.0 * field) / (dx ** 2)
        return field + self.diffusion_coeff * laplacian * dt
    
    def step(self, state, dt):
        exercise = state.get_organism_state('exercise_intensity', 0.0)
        sclerostin_field = state.get_field('bone', 'sclerostin')
        shape = sclerostin_field.shape
        
        # --- Compute current strain field from loading ---
        strain_field = self._compute_strain_field(shape, exercise)
        
        # --- Each osteocyte senses local strain and secretes sclerostin ---
        osteocyte_agents = state.get_agents('osteocytes')
        Kn = self.K_strain ** self.strain_cooperativity
        dt_min = dt / 60.0
        
        for agent in osteocyte_agents:
            i, j, k = agent['position']  # grid indices
            local_strain = strain_field[i, j, k]
            
            # Hill-type suppression: high strain → low secretion
            strain_n = local_strain ** self.strain_cooperativity
            suppression = Kn / (Kn + strain_n)
            
            # Per-agent mechanosensitivity modulates response
            secretion = (self.base_sclerostin_rate
                         * agent['state'].get('mechanosensitivity', 1.0)
                         * suppression
                         * dt_min)
            
            sclerostin_field[i, j, k] += secretion
        
        # --- Paracrine diffusion through bone matrix ---
        sclerostin_field = self._diffuse(sclerostin_field, dt)
        
        # --- First-order decay ---
        sclerostin_field *= np.exp(-self.decay_rate * dt)
        
        state.set_field('bone', 'sclerostin', sclerostin_field)


class OsteoblastBoneFormation_0030500(ProcessModel):
    """
    Osteoblast-mediated bone formation on spatial bone grid
    
    Mechanism: Osteoblasts deposit osteoid (collagen matrix) and mineralize
    it with hydroxyapatite (calcium-phosphate). Formation rate is regulated
    by circulating hormones (testosterone, growth hormone) and the local
    sclerostin concentration on the bone grid. Sclerostin antagonizes the
    Wnt/β-catenin pathway — low local sclerostin permits osteoblast
    activation, high local sclerostin suppresses it.
    
    Osteocalcin is secreted proportionally to formation activity into
    the circulation, serving as a serum biomarker and mild positive
    feedback signal.
    
    Mechanical loading effect on formation is captured INDIRECTLY:
    loading → osteocytes reduce sclerostin (SclerostinRegulation) →
    local sclerostin drops → osteoblasts here form more bone.
    This is the mechanistic basis of Wolff's law.
    
    Timescale: Hours (matrix deposition and mineralization)
    Location: Bone tissue (osteoblast surface, spatial)
    
    CSV rows covered:
        - osteoblast does osteocalcin secretion      [9, 38] ng/mL
        - osteoblast does collagen secretion          (implicit in formation_rate)
        - osteocalcin increases bone synthesis         (circulatory, positive feedback)
        - testosterone increases bone synthesis        (circulatory)
        - growth hormone increases bone synthesis      (circulatory)
        - mechanical loading increases bone synthesis  (INDIRECT via sclerostin field)
        - sclerostin decreases bone synthesis          (paracrine, read from local field)
    
    Equations:
        testosterone_effect  = testosterone / (K_test + testosterone)
        gh_effect            = GH / (K_gh + GH)
        osteocalcin_effect   = 0.5 + 0.5 × osteocalcin / (K_oc + osteocalcin)
        
        Per osteoblast at grid position (i,j,k):
            local_sclerostin = sclerostin_field[i,j,k]
            sclerostin_inhibition = 1 / (1 + local_sclerostin / K_scl)
            activity = test_eff × gh_eff × oc_eff × scl_inhib × agent.formation_capacity
            
            formation = base_rate × activity × dt
            calcium_store_field[i,j,k] += formation
            osteocalcin_delta += oc_secretion × activity × dt
    """
    
    inputs = {
        'testosterone':       ('blood', 'testosterone'),
        'growth_hormone':     ('blood', 'growth_hormone'),
        'osteocalcin':        ('blood', 'osteocalcin'),
        'sclerostin_field':   ('bone', 'sclerostin'),
        'calcium_store_field':('bone', 'calcium_store'),
    }
    outputs = {
        'calcium_store_field':('bone', 'calcium_store'),
        'osteocalcin':        ('blood', 'osteocalcin'),
    }
    
    parameters = {
        'base_formation_rate': {
            'default': 1e-7,
            'unit': 'relative/hr/agent',
            'range': (5e-8, 5e-7),
            'description': 'Per-osteoblast bone formation capacity at full activation'
        },
        'K_testosterone': {
            'default': 250.0,
            'unit': 'ng/dL',
            'range': (100.0, 400.0),
            'description': 'Androgen receptor sensitivity — testosterone for half-max formation stimulus'
        },
        'K_gh': {
            'default': 2.0,
            'unit': 'ng/mL',
            'range': (0.5, 5.0),
            'description': 'GH receptor sensitivity — growth hormone for half-max formation stimulus'
        },
        'K_sclerostin': {
            'default': 8.0,
            'unit': 'ng/mL',
            'range': (3.0, 15.0),
            'description': 'Wnt pathway sensitivity — sclerostin for half-max formation inhibition'
        },
        'K_osteocalcin': {
            'default': 15.0,
            'unit': 'ng/mL',
            'range': (5.0, 30.0),
            'description': 'Osteocalcin receptor sensitivity — half-max positive feedback'
        },
        'osteocalcin_secretion_rate': {
            'default': 0.08,
            'unit': 'ng/mL/hr/agent',
            'range': (0.02, 0.2),
            'description': 'Osteocalcin secreted per agent at full formation activity'
        },
    }
    
    def __init__(self, base_formation_rate=1e-7, K_testosterone=250.0,
                 K_gh=2.0, K_sclerostin=8.0, K_osteocalcin=15.0,
                 osteocalcin_secretion_rate=0.08):
        super().__init__("osteoblast_bone_formation", TimeScale.HOURS)
        self.base_formation_rate = base_formation_rate
        self.K_testosterone = K_testosterone
        self.K_gh = K_gh
        self.K_sclerostin = K_sclerostin
        self.K_osteocalcin = K_osteocalcin
        self.osteocalcin_secretion_rate = osteocalcin_secretion_rate
    
    def step(self, state, dt):
        testosterone = state.get_signal('blood', 'testosterone')
        gh = state.get_signal('blood', 'growth_hormone')
        osteocalcin = state.get_signal('blood', 'osteocalcin')
        sclerostin_field = state.get_field('bone', 'sclerostin')
        calcium_store_field = state.get_field('bone', 'calcium_store')
        
        osteoblast_agents = state.get_agents('osteoblasts')
        
        dt_hr = dt / 3600.0
        
        # --- Circulating hormone effects (global, same for all agents) ---
        testosterone_effect = testosterone / (self.K_testosterone + testosterone)
        gh_effect = gh / (self.K_gh + gh)
        
        # --- Osteocalcin positive feedback (bounded 0.5–1.0 to prevent runaway) ---
        osteocalcin_effect = 0.5 + 0.5 * osteocalcin / (self.K_osteocalcin + osteocalcin)
        
        global_modifiers = testosterone_effect * gh_effect * osteocalcin_effect
        
        total_osteocalcin = 0.0
        
        for agent in osteoblast_agents:
            i, j, k = agent['position']
            
            # --- Local sclerostin inhibition (spatial, Wolff's law endpoint) ---
            local_sclerostin = sclerostin_field[i, j, k]
            sclerostin_inhibition = 1.0 / (1.0 + local_sclerostin / self.K_sclerostin)
            
            # --- Combined activity: global hormones × local sclerostin × agent state ---
            activity = (global_modifiers * sclerostin_inhibition
                        * agent['state'].get('formation_capacity', 1.0))
            
            # --- Deposit mineral locally on grid ---
            formation = self.base_formation_rate * activity * dt_hr
            calcium_store_field[i, j, k] += formation
            
            # --- Osteocalcin secretion into circulation ---
            oc_secreted = self.osteocalcin_secretion_rate * activity * dt_hr
            total_osteocalcin += oc_secreted
        
        state.set_field('bone', 'calcium_store', calcium_store_field)
        state.update_signal('blood', 'osteocalcin', total_osteocalcin)


class OsteoclastBoneResorption_GO0046850(ProcessModel):
    """
    UPDATED — replaces the version in skeletal_additions.py.
 
    Adds three new modulators:
      calcitriol → VDR-mediated osteoclast suppression
      OPG        → RANKL sequestration, inhibits osteoclastogenesis
      inflammation → cytokine (TNF-α/IL-1) driven osteoclast activation
 
    PDF rows newly encoded (on top of original rows):
    - calcitriol decreases bone calcium secretion (VDR on osteoclast precursors)
    - osteoprotegrin decreases bone degradation (OPG/RANKL ratio)
    - inflammation increases bone degradation (cytokine RANKL upregulation)
    """
    inputs = {
        'pth':          ('blood', 'parathyroid_hormone'),
        'estrogen':     ('blood', 'estrogen'),
        't3':           ('blood', 't3'),
        'calcitriol':   ('blood', 'calcitriol'),
        'opg':          ('blood', 'opg'),
        'inflammation': ('blood', 'inflammation'),
        'calcium_store':('bone',  'calcium_store'),
    }
    outputs = {
        'calcium_store':  ('bone',  'calcium_store'),
        'blood_calcium':  ('blood', 'calcium'),
        'blood_phosphate':('blood', 'phosphate'),
    }
    parameters = {
        'base_resorption_rate':   {'default': 5e-8,  'unit': 'relative/hr/agent'},
        'K_pth':                  {'default': 30.0,  'unit': 'pg/mL'},
        'K_estrogen':             {'default': 0.05,  'unit': 'ng/mL'},
        'K_t3':                   {'default': 1.0,   'unit': 'ng/mL'},
        'K_calcitriol':           {'default': 40.0,  'unit': 'pg/mL'},
        'K_opg':                  {'default': 0.8,   'unit': 'relative'},
        'max_calcitriol_inhib':   {'default': 0.35,  'unit': 'dimensionless'},
        'max_opg_inhib':          {'default': 0.40,  'unit': 'dimensionless'},
        'max_inflam_activation':  {'default': 1.5,   'unit': 'dimensionless'},
        'calcium_per_unit':       {'default': 350.0, 'unit': 'mg/relative'},
        'phosphate_calcium_ratio':{'default': 0.63,  'unit': 'mg/mg'},
        'blood_volume_dL':        {'default': 50.0,  'unit': 'dL'},
    }
 
    def __init__(self, base_resorption_rate=5e-8, K_pth=30.0, K_estrogen=0.05,
                 K_t3=1.0, K_calcitriol=40.0, K_opg=0.8,
                 max_calcitriol_inhib=0.35, max_opg_inhib=0.40,
                 max_inflam_activation=1.5, calcium_per_unit=350.0,
                 phosphate_calcium_ratio=0.63, blood_volume_dL=50.0):
        super().__init__("osteoclast_bone_resorption", TimeScale.HOURS)
        self.base_resorption_rate    = base_resorption_rate
        self.K_pth                   = K_pth
        self.K_estrogen              = K_estrogen
        self.K_t3                    = K_t3
        self.K_calcitriol            = K_calcitriol
        self.K_opg                   = K_opg
        self.max_calcitriol_inhib    = max_calcitriol_inhib
        self.max_opg_inhib           = max_opg_inhib
        self.max_inflam_activation   = max_inflam_activation
        self.calcium_per_unit        = calcium_per_unit
        self.phosphate_calcium_ratio = phosphate_calcium_ratio
        self.blood_volume_dL         = blood_volume_dL
 
    def step(self, state, dt):
        pth        = state.get_signal('blood', 'parathyroid_hormone')
        estrogen   = state.get_signal('blood', 'estrogen')
        t3         = state.get_signal('blood', 't3')
        calcitriol = state.get_signal('blood', 'calcitriol')
        opg        = state.get_signal('blood', 'opg')
        inflam     = state.get_signal('blood', 'inflammation')
        ca_field   = state.get_field('bone', 'calcium_store')
 
        dt_hr = dt / 3600.0
 
        # --- Activators ---
        pth_act   = pth / (self.K_pth + pth)
        t3_act    = 1.0 + t3 / (self.K_t3 + t3)
        inflam_act= 1.0 + self.max_inflam_activation * inflam
 
        # --- Inhibitors ---
        estrogen_inh   = 0.5 * estrogen   / (self.K_estrogen + estrogen)
        calcitriol_inh = self.max_calcitriol_inhib * calcitriol / (self.K_calcitriol + calcitriol)
        opg_inh        = self.max_opg_inhib        * opg        / (self.K_opg        + opg)
 
        activity = (pth_act * t3_act * inflam_act
                    * (1.0 - estrogen_inh)
                    * (1.0 - calcitriol_inh)
                    * (1.0 - opg_inh))
 
        total_resorbed = 0.0
        for agent in state.get_agents('osteoclasts'):
            i, j, k     = agent['position']
            local_store = max(ca_field[i, j, k], 0.0)
            resorption  = (self.base_resorption_rate
                           * activity
                           * agent['state'].get('resorption_capacity', 1.0)
                           * local_store * dt_hr)
            ca_field[i, j, k] -= resorption
            total_resorbed     += resorption
 
        ca_rel  = total_resorbed * self.calcium_per_unit
        po4_rel = ca_rel * self.phosphate_calcium_ratio
 
        state.set_field('bone', 'calcium_store', ca_field)
        state.update_signal('blood', 'calcium',   ca_rel  / self.blood_volume_dL)
        state.update_signal('blood', 'phosphate', po4_rel / self.blood_volume_dL)
 
 
# ===========================================================================
# NEW 2 ─ RenalCalciumFiltration_GO0070293
# ===========================================================================
class RenalCalciumFiltration_GO0070293(ProcessModel):
    """
    Glomerular filtration followed by immediate passive PCT and TAL reabsorption.
 
    Only ionised (free) calcium is filtered (~60% of plasma Ca; 40% is protein-bound).
    PCT reabsorbs ~65% passively (paracellular, electrochemical gradient).
    TAL reabsorbs ~20% passively (NKCC2-driven lumen positive potential).
    The remaining ~15% (DCT fraction) is stored in kidney.filtered_calcium for
    downstream PTH/VitD-regulated reabsorption (DCTCalciumReabsorption_0035898)
    and collecting duct clearance (RenalCalciumExcretion_GO0070293).
 
    Total reabsorption: PCT 65% + TAL 20% + DCT ~9% + CD ~5% = 99% ≈ physiology.
 
    PDF rows encoded:
    ┌──────────────────────────────────────────────────────────────────────────┐
    │ urinary  │ kidney does Ca²⁺ reabsorption to blood, 98–99% of filtered   │
    │ urinary  │ PCT does Ca²⁺ reabsorption, 65% (passive paracellular)       │
    │ urinary  │ TAL does Ca²⁺ reabsorption, 20%                              │
    │ urinary  │ Ca²⁺ excretion = GFR × [Ca²⁺]plasma − Ca²⁺ reabsorbed      │
    │ urinary  │ blood flow into glomerulus increases Ca²⁺ excretion          │
    │ urinary  │ plasma Ca²⁺ decreases Ca²⁺ reabsorption                     │
    └──────────────────────────────────────────────────────────────────────────┘
    """
 
    inputs  = {'blood_calcium': ('blood', 'calcium')}
    outputs = {
        'filtered_calcium': ('kidney', 'filtered_calcium'),
        'blood_calcium':    ('blood',  'calcium'),
    }
 
    parameters = {
        'gfr':               {'default': 120.0, 'unit': 'mL/min'},
        'ionized_fraction':  {'default': 0.60,  'unit': 'fraction',
                              'description': '60% free ionised; 40% albumin-bound'},
        'pct_reabsorption':  {'default': 0.65,  'unit': 'fraction'},
        'tal_reabsorption':  {'default': 0.20,  'unit': 'fraction'},
        'blood_volume_dL':   {'default': 50.0,  'unit': 'dL'},
    }
 
    def __init__(self, gfr=120.0, ionized_fraction=0.60, pct_reabsorption=0.65,
                 tal_reabsorption=0.20, blood_volume_dL=50.0):
        super().__init__("renal_calcium_filtration", TimeScale.MINUTES)
        self.gfr              = gfr
        self.ionized_fraction = ionized_fraction
        self.pct_reabsorption = pct_reabsorption
        self.tal_reabsorption = tal_reabsorption
        self.blood_volume_dL  = blood_volume_dL
 
    def step(self, state, dt):
        ca = state.get_signal('blood', 'calcium')
 
        # Glomerular filtration (mg/min): only free ionised fraction is filtered
        # GFR [mL/min] × ionised_Ca [mg/dL] × (1 dL / 100 mL) = mg/min
        filtration_rate = self.gfr * self.ionized_fraction * ca / 100.0   # mg/min
        filtered        = filtration_rate * (dt / 60.0)                   # mg this step
 
        # PCT (passive paracellular, ~65%) and TAL (~20%) reclaim immediately
        pct_returned = filtered * self.pct_reabsorption
        tal_returned = filtered * self.tal_reabsorption
        to_dct       = filtered - pct_returned - tal_returned             # ~15% of filtered
 
        # Net blood calcium change: lose only the DCT-bound fraction
        # (PCT + TAL return is instantaneous within the same step)
        state.update_signal('blood', 'calcium',   -to_dct / self.blood_volume_dL)
 
        # Overwrite kidney.filtered_calcium each minute (it is consumed by DCT + excretion)
        state.set_signal('kidney', 'filtered_calcium', max(to_dct, 0.0))
 
 
# ===========================================================================
# NEW 3 ─ RenalCalciumExcretion_GO0070293
# ===========================================================================
class RenalCalciumExcretion_GO0070293(ProcessModel):
    """
    Collecting duct (CD) calcium reabsorption and final urinary excretion.
 
    Runs AFTER RenalCalciumFiltration_GO0070293 and DCTCalciumReabsorption_0035898.
    The CD reabsorbs ~5% of total filtered (≈ 33% of what remains after DCT).
    Whatever remains exits as urinary calcium.
 
    PDF rows encoded:
    ┌──────────────────────────────────────────────────────────────────────────┐
    │ urinary  │ CD does Ca²⁺ reabsorption, 5% of total filtered             │
    │ urinary  │ kidney does Ca²⁺ excretion to urine, ~20% of daily intake   │
    │ urinary  │ Ca²⁺ excretion = filtered − reabsorbed                      │
    └──────────────────────────────────────────────────────────────────────────┘
    """
 
    inputs  = {'filtered_calcium': ('kidney', 'filtered_calcium')}
    outputs = {
        'urinary_calcium':  ('kidney', 'urinary_calcium'),
        'filtered_calcium': ('kidney', 'filtered_calcium'),
        'blood_calcium':    ('blood',  'calcium'),
    }
 
    parameters = {
        'cd_reabsorption_fraction': {
            'default': 0.33, 'unit': 'fraction of DCT input',
            'description': '~5% of total filtered = ~33% of the 15% reaching DCT+CD'
        },
        'blood_volume_dL': {'default': 50.0, 'unit': 'dL'},
    }
 
    def __init__(self, cd_reabsorption_fraction=0.33, blood_volume_dL=50.0):
        super().__init__("renal_calcium_excretion", TimeScale.MINUTES)
        self.cd_reabsorption_fraction = cd_reabsorption_fraction
        self.blood_volume_dL          = blood_volume_dL
 
    def step(self, state, dt):
        remaining = state.get_signal('kidney', 'filtered_calcium')    # mg remaining after DCT
 
        if remaining <= 0.0:
            return
 
        # CD passive reabsorption (~5% total filtered, VitD-responsive but approximated constant)
        cd_reabsorbed = remaining * self.cd_reabsorption_fraction
        urinary       = remaining - cd_reabsorbed
 
        # CD reclaims Ca back to blood
        state.update_signal('blood', 'calcium',     cd_reabsorbed / self.blood_volume_dL)
 
        # Record urinary loss and clear the filtered pool
        state.set_signal('kidney', 'urinary_calcium',  max(urinary, 0.0))
        state.set_signal('kidney', 'filtered_calcium', 0.0)
 
 
# ===========================================================================
# NEW 4 ─ IntestinalCalciumAbsorption_GO0055074
# ===========================================================================
class IntestinalCalciumAbsorption_GO0055074(ProcessModel):
    """
    Vitamin D-dependent intestinal calcium absorption (duodenum / proximal jejunum).
 
    Mechanism:
    • Active transcellular (~15% baseline): calcitriol induces TRPV6 (apical entry)
      and calbindin-D9k (intracellular shuttle) → Ca²⁺ exits via PMCA / NCX.
    • Passive paracellular (~5% baseline): driven by electrochemical gradient through
      tight junctions; relatively insensitive to hormones.
    Total baseline ≈ 20%.  Calcitonin reduces transcellular component.
    PTH effect is largely indirect (via calcitriol production); omitted here to
    avoid double-counting with ProximalTubuleHydroxylase.
 
    PDF rows encoded:
    ┌──────────────────────────────────────────────────────────────────────────┐
    │ digestive │ enterocytes do Ca²⁺ transport to blood, 20% of intake       │
    │ digestive │ Vitamin D increases Ca²⁺ transport (TRPV6 / CaBP induction) │
    │ digestive │ PTH increases Ca²⁺ transport (indirect via calcitriol)       │
    │ digestive │ calcitonin decreases Ca²⁺ transport                          │
    │ digestive │ passive paracellular Ca²⁺ absorption (tight junctions)       │
    │ digestive │ enterocytes do Ca²⁺ excretion to lumen, 80% (net)           │
    └──────────────────────────────────────────────────────────────────────────┘
 
    Rate:
        absorbed_fraction = base × vitd_factor × calcitonin_factor
        vitd_factor       = 0.30 + 1.40 × [D3] / (K_D3 + [D3])
          → ≈ 1.08 at normal [D3]=50 pg/mL;  0.30 at [D3]=0 (passive only)
        calcitonin_factor = 1 / (1 + [CT] / K_CT)
        absorbed (mg/step) = dietary_Ca × fraction × dt / 86400
    """
 
    inputs  = {
        'calcitriol':  ('blood', 'calcitriol'),
        'calcitonin':  ('blood', 'calcitonin'),
    }
    outputs = {'blood_calcium': ('blood', 'calcium')}
 
    parameters = {
        'base_absorption_fraction': {
            'default': 0.20, 'unit': 'fraction',
            'description': 'Baseline fractional absorption; 15% active + 5% passive'
        },
        'K_calcitriol': {
            'default': 40.0, 'unit': 'pg/mL',
            'description': 'Calcitriol for half-max TRPV6/calbindin induction'
        },
        'K_calcitonin': {
            'default': 50.0, 'unit': 'same units as blood.calcitonin in anatomy.yaml',
            'description': 'Calcitonin for half-max transcellular inhibition'
        },
        'blood_volume_dL': {'default': 50.0, 'unit': 'dL'},
    }
 
    def __init__(self, base_absorption_fraction=0.20, K_calcitriol=40.0,
                 K_calcitonin=50.0, blood_volume_dL=50.0):
        super().__init__("intestinal_calcium_absorption", TimeScale.HOURS)
        self.base_absorption_fraction = base_absorption_fraction
        self.K_calcitriol             = K_calcitriol
        self.K_calcitonin             = K_calcitonin
        self.blood_volume_dL          = blood_volume_dL
 
    def step(self, state, dt):
        calcitriol = state.get_signal('blood', 'calcitriol')
        calcitonin = state.get_signal('blood', 'calcitonin')
        dietary_ca = state.get_organism_state('dietary_calcium', 1000.0)  # mg/day
 
        # VitD: upregulates TRPV6 and calbindin-D9k (active transcellular route)
        # Factor ranges 0.30 (VitD deficient) → ~1.47 (high VitD)
        vitd_factor = 0.30 + 1.40 * calcitriol / (self.K_calcitriol + calcitriol)
 
        # Calcitonin: inhibits intestinal Ca absorption
        calcitonin_factor = 1.0 / (1.0 + calcitonin / self.K_calcitonin)
 
        # Effective absorption fraction
        frac = self.base_absorption_fraction * vitd_factor * calcitonin_factor
        frac = float(np.clip(frac, 0.05, 0.60))
 
        # Calcium absorbed this step (dietary_ca mg/day → mg/step)
        absorbed = dietary_ca * frac * (dt / 86400.0)
        state.update_signal('blood', 'calcium', absorbed / self.blood_volume_dL)
 
 
# ===========================================================================
# NEW 5 ─ IntestinalPhosphateAbsorption_GO0006817
# ===========================================================================
class IntestinalPhosphateAbsorption_GO0006817(ProcessModel):
    """
    Intestinal phosphate absorption via NaPi-IIb cotransporter (SLC34A2).
 
    Mechanism: NaPi-IIb mediates active Na⁺-coupled PO₄³⁻ uptake in the proximal
    small intestine.  Calcitriol up-regulates NaPi-IIb transcription.  FGF23 exerts
    a direct (minor) suppressive effect on intestinal NaPi-IIb, independent of its
    calcitriol-suppressive effect (which is captured by VitaminDCatabolism_ODE and
    ProximalTubuleHydroxylase and therefore NOT double-counted here).
    Passive paracellular diffusion provides a concentration-dependent baseline.
 
    PDF rows encoded:
    ┌──────────────────────────────────────────────────────────────────────────┐
    │ digestive │ NaPi-IIb mediates PO₄³⁻ reabsorption, 73% of intake        │
    │ digestive │ Vitamin D increases PO₄³⁻ transport (NaPi-IIb induction)    │
    │ digestive │ FGF23 decreases PO₄³⁻ transport (direct intestinal effect)  │
    │ digestive │ concentration-dependent passive diffusion                    │
    │ digestive │ ~73% of phosphorus intake absorbed                           │
    │ digestive │ pancreatic/intestinal secretions → PO₄ to faeces (~13%)     │
    └──────────────────────────────────────────────────────────────────────────┘
 
    Rate:
        frac  = base × vitd_factor × fgf23_factor
        vitd_factor  = 0.50 + [D3] / (K_D3 + [D3])
        fgf23_factor = 1 − 0.30 × FGF23 / (K_FGF23 + FGF23)
        absorbed = dietary_phos × frac × dt / 86400
    """
 
    inputs  = {
        'calcitriol': ('blood', 'calcitriol'),
        'fgf23':      ('blood', 'fgf23'),
    }
    outputs = {'blood_phosphate': ('blood', 'phosphate')}
 
    parameters = {
        'dietary_phosphate': {
            'default': 1000.0, 'unit': 'mg/day',
            'description': 'Typical Western dietary phosphate intake'
        },
        'base_absorption_fraction': {
            'default': 0.73, 'unit': 'fraction',
            'description': '73% of intake absorbed under normal conditions'
        },
        'K_calcitriol': {
            'default': 40.0, 'unit': 'pg/mL',
            'description': 'Calcitriol for half-max NaPi-IIb induction'
        },
        'K_fgf23': {
            'default': 150.0, 'unit': 'RU/mL',
            'description': 'FGF23 for half-max direct intestinal suppression '
                           '(higher than renal K_FGF23 — intestinal effect is weaker)'
        },
        'blood_volume_dL': {'default': 50.0, 'unit': 'dL'},
    }
 
    def __init__(self, dietary_phosphate=1000.0, base_absorption_fraction=0.73,
                 K_calcitriol=40.0, K_fgf23=150.0, blood_volume_dL=50.0):
        super().__init__("intestinal_phosphate_absorption", TimeScale.HOURS)
        self.dietary_phosphate         = dietary_phosphate
        self.base_absorption_fraction  = base_absorption_fraction
        self.K_calcitriol              = K_calcitriol
        self.K_fgf23                   = K_fgf23
        self.blood_volume_dL           = blood_volume_dL
 
    def step(self, state, dt):
        calcitriol = state.get_signal('blood', 'calcitriol')
        fgf23      = state.get_signal('blood', 'fgf23')
 
        # VitD (calcitriol) up-regulates NaPi-IIb → increases active transport
        # Factor: 0.50 at [D3]=0 → ~1.0 at normal [D3]=50 pg/mL
        vitd_factor = 0.50 + calcitriol / (self.K_calcitriol + calcitriol)
 
        # FGF23 direct intestinal suppression (minor; K_FGF23 set high to reflect this)
        fgf23_factor = 1.0 - 0.30 * fgf23 / (self.K_fgf23 + fgf23)
 
        frac = self.base_absorption_fraction * vitd_factor * fgf23_factor
        frac = float(np.clip(frac, 0.10, 0.95))
 
        # Phosphate absorbed this step
        absorbed = self.dietary_phosphate * frac * (dt / 86400.0)      # mg per step
        state.update_signal('blood', 'phosphate', absorbed / self.blood_volume_dL)
 

  
# ===========================================================================
# STUB 1 ─ PhosphateBalance_ODE_GO0006817
# ===========================================================================
class PhosphateBalance_ODE_GO0006817(ProcessModel):
    """
    Blood phosphate pool — soft homeostatic restoration.
 
    Does NOT double-count intestinal input or renal excretion; those are handled
    by IntestinalPhosphateAbsorption_GO0006817 and RenalPhosphateExcretion_ODE_GO0070293.
    This model represents the aggregate of soft-tissue uptake/release, bone exchange,
    and intracellular buffering — lumped as a first-order drive toward the set-point.
 
    PDF rows encoded:
    ┌──────────────────────────────────────────────────────────────────────────┐
    │ urinary / digestive  │ kidney does PO4 reabsorption to blood, 80%       │
    │                      │ kidney does PO4 excretion to urine, 60% of intake│
    │                      │ intestinal PO4 absorption via NaPi-IIb (basal)   │
    └──────────────────────────────────────────────────────────────────────────┘
 
    ODE:
        d[Phos]/dt = (basal_phos − [Phos]) / τ
        τ = 7200 s  (2-hour soft correction; represents aggregate buffering)
    """
 
    inputs  = {}  # no required signal inputs; purely driven by set-point
    outputs = {'blood_phosphate': ('blood', 'phosphate')}
 
    parameters = {
        'blood_volume': {
            'default': 3000.0, 'unit': 'mL',
            'description': 'Effective ECF + plasma distribution volume for phosphate (~30 dL)'
        },
        'basal_phos': {
            'default': 3.7, 'unit': 'mg/dL',
            'description': 'Homeostatic set-point (normal serum phosphate 3.5–4.5 mg/dL)'
        },
    }
 
    def __init__(self, blood_volume=3000.0, basal_phos=3.7):
        super().__init__("phosphate_balance", TimeScale.MINUTES)
        self.blood_volume = blood_volume          # mL
        self.basal_phos   = basal_phos            # mg/dL
        self._tau         = 7200.0                # s — 2-hour equilibration
 
    def step(self, state, dt):
        phos = state.get_signal('blood', 'phosphate')
 
        # Soft first-order restoration toward set-point
        delta = (self.basal_phos - phos) * (dt / self._tau)
        state.update_signal('blood', 'phosphate', delta)
 
 
# ===========================================================================
# STUB 2 ─ FGF23Secretion_ODE_GO0006817
# ===========================================================================
class FGF23Secretion_ODE_GO0006817(ProcessModel):
    """
    FGF23 secretion by osteocytes — regulated by phosphate and calcitriol.
 
    Mechanism: elevated serum phosphate and calcitriol both up-regulate FGFR1/3-Klotho
    signalling on osteocytes, increasing FGF23 gene transcription.  FGF23 is then
    cleared with a half-life of ~50–60 min.  PTH has a secondary stimulatory effect
    via increased phosphate load (modelled here implicitly through [Phos]).
 
    PDF rows encoded:
    ┌──────────────────────────────────────────────────────────────────────────┐
    │ skeletal  │ osteocyte does FGF23 secretion                              │
    │ skeletal  │ parathyroid hormone increases FGF23 secretion               │
    │ skeletal  │ calcitriol increases FGF23 secretion                        │
    └──────────────────────────────────────────────────────────────────────────┘
 
    ODE:
        F_target  = basal × (1 + phos_sensitivity × max(0, [P]−P_thresh) / P_thresh)
                          × max(0.3, [D3] / D3_normal)
        production = clearance × F_target        (sets FGF23_ss = F_target)
        d[FGF23]/dt = production − clearance × [FGF23]
 
        The internal lag state (_prod) smooths production changes
        with time_constant τ_prod = time_constant (min).
    """
 
    inputs  = {
        'phosphate':  ('blood', 'phosphate'),
        'calcitriol': ('blood', 'calcitriol'),
    }
    outputs = {'fgf23': ('blood', 'fgf23')}
 
    parameters = {
        'basal': {
            'default': 40.0, 'unit': 'RU/mL',
            'description': 'FGF23 set-point at normal phosphate and calcitriol'
        },
        'phos_sensitivity': {
            'default': 10.0, 'unit': 'dimensionless',
            'description': 'Fold-increase in FGF23 per unit fractional phosphate excess'
        },
        'phos_threshold': {
            'default': 3.5, 'unit': 'mg/dL',
            'description': 'Phosphate level at which FGF23 secretion begins to rise'
        },
        'time_constant': {
            'default': 120.0, 'unit': 'min',
            'description': 'Lag between phosphate rise and FGF23 production increase'
        },
        'clearance': {
            'default': 0.01, 'unit': '/min',
            'description': 'FGF23 clearance rate constant (half-life ≈ 69 min; physiological ≈ 50–70 min)'
        },
    }
 
    def __init__(self, basal=40.0, phos_sensitivity=10.0, phos_threshold=3.5,
                 time_constant=120.0, clearance=0.01):
        super().__init__("fgf23_secretion", TimeScale.MINUTES)
        self.basal            = basal
        self.phos_sensitivity = phos_sensitivity
        self.phos_threshold   = phos_threshold
        self.time_constant    = time_constant   # min
        self.clearance        = clearance       # /min
 
        # Internal production rate — adapts to stimulus with lag
        self._prod = clearance * basal  # initial: at steady state
 
    def step(self, state, dt):
        fgf23     = state.get_signal('blood', 'fgf23')
        phos      = state.get_signal('blood', 'phosphate')
        calcitriol = state.get_signal('blood', 'calcitriol')
 
        calcitriol_normal = 50.0  # pg/mL
 
        # Phosphate-driven stimulation (linear above threshold)
        phos_excess = max(0.0, phos - self.phos_threshold) / self.phos_threshold
        phos_factor = 1.0 + self.phos_sensitivity * phos_excess
 
        # Calcitriol stimulation (positive feedback; clamped at 0.3 to prevent full shutdown)
        calcitriol_factor = max(0.3, calcitriol / calcitriol_normal)
 
        # Stimulus-dependent production target
        F_target     = self.basal * phos_factor * calcitriol_factor
        prod_target  = self.clearance * F_target
 
        # Lagged production ODE: d(prod)/dt = (prod_target − prod) / τ
        dt_min      = dt / 60.0
        self._prod += (prod_target - self._prod) * (dt_min / self.time_constant)
 
        # FGF23 pool ODE: d[FGF23]/dt = production − clearance × [FGF23]
        dFGF23 = (self._prod - self.clearance * fgf23) * dt_min
        state.update_signal('blood', 'fgf23', dFGF23)
 
 
# ===========================================================================
# STUB 3 ─ VitaminDCatabolism_ODE_GO0030500
# ===========================================================================
class VitaminDCatabolism_ODE_GO0030500(ProcessModel):
    """
    24-hydroxylase (CYP24A1) catabolism of calcitriol.
 
    Mechanism: calcitriol auto-induces CYP24A1 (negative feedback).  FGF23 also
    induces CYP24A1 via FGFR1/Klotho signalling.  PTH mildly suppresses CYP24A1
    (protecting calcitriol while PTH is elevated — ensures vitamin D effect persists
    when calcium is low).
 
    PDF rows encoded:
    ┌──────────────────────────────────────────────────────────────────────────┐
    │ endocrine │ FGF23 decreases calcifediol → calcitriol conversion         │
    │ skeletal  │ FGF23 decreases calcifediol → calcitriol (indirectly via    │
    │           │   24-hydroxylase induction)                                 │
    │ endocrine │ PTH decreases (suppresses 24-OH in context of production)   │
    └──────────────────────────────────────────────────────────────────────────┘
 
    ODE:
        enzyme = basal_inactivation
               + calcitriol_feedback × ([D3] / D3_normal)
               + fgf23_enhancement  × ([FGF23] / FGF23_normal)
        enzyme *= (1 − pth_suppression × PTH / (K_pth + PTH))
        d[D3]/dt = −enzyme × [D3]        (first-order degradation)
    """
 
    inputs  = {
        'calcitriol':        ('blood', 'calcitriol'),
        'fgf23':             ('blood', 'fgf23'),
        'parathyroid_hormone': ('blood', 'parathyroid_hormone'),
    }
    outputs = {'calcitriol': ('blood', 'calcitriol')}
 
    parameters = {
        'basal_inactivation': {
            'default': 0.008, 'unit': '/min',
            'description': 'Basal CYP24A1 first-order degradation rate constant'
        },
        'calcitriol_feedback': {
            'default': 0.02, 'unit': '/min per normalised D3',
            'description': 'Calcitriol auto-induction of CYP24A1'
        },
        'fgf23_enhancement': {
            'default': 0.05, 'unit': '/min per normalised FGF23',
            'description': 'FGF23 induction of CYP24A1 via FGFR1/Klotho'
        },
        'pth_suppression': {
            'default': 0.3, 'unit': 'dimensionless',
            'description': 'Maximum fractional CYP24A1 suppression by PTH'
        },
    }
 
    _K_PTH            = 30.0   # pg/mL — PTH for half-max CYP24A1 suppression
    _CALCITRIOL_NORM  = 50.0   # pg/mL
    _FGF23_NORM       = 30.0   # RU/mL
 
    def __init__(self, basal_inactivation=0.008, calcitriol_feedback=0.02,
                 fgf23_enhancement=0.05, pth_suppression=0.3):
        super().__init__("vitamin_d_catabolism", TimeScale.MINUTES)
        self.basal_inactivation  = basal_inactivation
        self.calcitriol_feedback = calcitriol_feedback
        self.fgf23_enhancement   = fgf23_enhancement
        self.pth_suppression     = pth_suppression
 
    def step(self, state, dt):
        calcitriol = state.get_signal('blood', 'calcitriol')
        fgf23      = state.get_signal('blood', 'fgf23')
        pth        = state.get_signal('blood', 'parathyroid_hormone')
 
        # CYP24A1 activity: basal + calcitriol auto-induction + FGF23 induction
        enzyme = (self.basal_inactivation
                  + self.calcitriol_feedback * (calcitriol / self._CALCITRIOL_NORM)
                  + self.fgf23_enhancement   * (fgf23      / self._FGF23_NORM))
 
        # PTH suppresses CYP24A1 (Hill function)
        pth_sup = self.pth_suppression * pth / (self._K_PTH + pth)
        enzyme  = max(0.0, enzyme * (1.0 - pth_sup))
 
        # First-order degradation over step
        degraded = enzyme * calcitriol * (dt / 60.0)                # pg/mL removed
        degraded = min(degraded, calcitriol * 0.5)                   # safety: ≤50% per step
        state.update_signal('blood', 'calcitriol', -degraded)
 
 
# ===========================================================================
# STUB 4 ─ RenalPhosphateExcretion_ODE_GO0070293
# ===========================================================================
class RenalPhosphateExcretion_ODE_GO0070293(ProcessModel):
    """
    PCT phosphate reabsorption regulated by PTH and FGF23.
 
    Mechanism: NaPi-II cotransporters (NaPi-IIa/c) are internalised and degraded
    in response to PTH (via cAMP) or FGF23 (via MAPK/ERK).  Both hormones reduce
    the reabsorption fraction → phosphaturia.  Calcitriol mildly increases NaPi-IIb
    expression → mild anti-phosphaturic effect.
 
    PDF rows encoded:
    ┌──────────────────────────────────────────────────────────────────────────┐
    │ urinary   │ PCT does PO4 reabsorption, 85% of filtered load             │
    │ urinary   │ PTH decreases PO4 reabsorption (phosphaturia)               │
    │ urinary   │ FGF23 increases PO4 export to urine                         │
    │ urinary   │ FGF23 decreases PCT NaPi-IIa reabsorption                   │
    │ urinary   │ Vitamin D increases PO4 reabsorption                        │
    │ urinary   │ low pH decreases PO4 reabsorption (implicit in basal)       │
    └──────────────────────────────────────────────────────────────────────────┘
 
    Model:
        filtered_rate = GFR × 0.90 × [Phos] / 100          (mg/min)
        Δreabsorption = basal − PTH_inhibition − FGF23_inhibition + calcitriol_boost
        excreted_rate = filtered_rate × (1 − Δreabsorption) (mg/min)
        d[Phos]/dt   −= excreted_rate / V_blood
    """
 
    inputs  = {
        'phosphate':           ('blood', 'phosphate'),
        'parathyroid_hormone': ('blood', 'parathyroid_hormone'),
        'fgf23':               ('blood', 'fgf23'),
        'calcitriol':          ('blood', 'calcitriol'),
    }
    outputs = {'phosphate': ('blood', 'phosphate')}
 
    parameters = {
        'basal_reabsorption': {
            'default': 0.80, 'unit': 'fraction',
            'description': 'Non-hormonal PCT reabsorption fraction (80–85% physiological range)'
        },
        'gfr': {
            'default': 120.0, 'unit': 'mL/min'
        },
        'pth_inhibition_gain': {
            'default': 0.8, 'unit': 'dimensionless',
            'description': 'Maximum PTH-driven fractional reduction in reabsorption'
        },
        'fgf23_inhibition_gain': {
            'default': 1.0, 'unit': 'dimensionless',
            'description': 'Maximum FGF23-driven fractional reduction in reabsorption'
        },
    }
 
    _K_PTH        = 30.0    # pg/mL
    _K_FGF23      = 40.0    # RU/mL
    _K_CALCITRIOL = 40.0    # pg/mL
    _FREE_FILT    = 0.90    # freely filtered fraction of plasma phosphate
    _V_BLOOD_DL   = 50.0    # 5 L blood = 50 dL
 
    def __init__(self, basal_reabsorption=0.80, gfr=120.0,
                 pth_inhibition_gain=0.8, fgf23_inhibition_gain=1.0):
        super().__init__("renal_phosphate_excretion", TimeScale.MINUTES)
        self.basal_reabsorption    = basal_reabsorption
        self.gfr                   = gfr
        self.pth_inhibition_gain   = pth_inhibition_gain
        self.fgf23_inhibition_gain = fgf23_inhibition_gain
 
    def step(self, state, dt):
        phos      = state.get_signal('blood', 'phosphate')
        pth       = state.get_signal('blood', 'parathyroid_hormone')
        fgf23     = state.get_signal('blood', 'fgf23')
        calcitriol = state.get_signal('blood', 'calcitriol')
 
        # Glomerular filtration (rate in mg/min)
        filtered_rate = self.gfr * self._FREE_FILT * (phos / 100.0)
 
        # Hormonal modulation of NaPi-II cotransporter expression
        pth_inhibit   = self.pth_inhibition_gain   * pth   / (self._K_PTH   + pth)
        fgf23_inhibit = self.fgf23_inhibition_gain * fgf23 / (self._K_FGF23 + fgf23)
        vit_d_boost   = 0.10 * calcitriol          / (self._K_CALCITRIOL + calcitriol)
 
        reabs = self.basal_reabsorption - pth_inhibit - fgf23_inhibit + vit_d_boost
        reabs = float(np.clip(reabs, 0.05, 0.95))
 
        excretion_rate = filtered_rate * (1.0 - reabs)                   # mg/min
        excreted       = excretion_rate * (dt / 60.0)                    # mg per step
 
        state.update_signal('blood', 'phosphate', -excreted / self._V_BLOOD_DL)
 
 
# ===========================================================================
# STUB 5 ─ EstrogenBoneProtection_ODE_GO0046850
# ===========================================================================
class EstrogenBoneProtection_ODE_GO0046850(ProcessModel):
    """
    Estrogen dynamics ODE.  The bone-protection effect (osteoclast suppression)
    is exerted through blood.estrogen, which OsteoclastBoneResorption_GO0046850 reads.
 
    Mechanism: estrogen → OPG (osteoprotegerin) up-regulation + RANKL down-regulation
    on osteoblasts/osteocytes → reduces RANK signalling on osteoclasts → inhibits
    osteoclast differentiation and accelerates osteoclast apoptosis.
 
    PDF rows encoded:
    ┌──────────────────────────────────────────────────────────────────────────┐
    │ skeletal  │ estrogen decreases bone degradation (osteoclast)             │
    │ skeletal  │ osteoprotegrin decreases RANKL concentration                 │
    │ skeletal  │ osteoprotegrin decreases bone degradation                    │
    └──────────────────────────────────────────────────────────────────────────┘
 
    ODE:
        d[E]/dt = clearance_rate × (basal_estrogen − [E])
        (first-order toward basal; time_constant = 1/clearance_rate)
    """
 
    inputs  = {}
    outputs = {'estrogen': ('blood', 'estrogen')}
 
    parameters = {
        'basal_estrogen': {
            'default': 0.2, 'unit': 'ng/mL (or activity units)',
            'description': 'Steady-state estrogen set-point; tune for sex/age'
        },
        'reproductive_estrogen': {
            'default': 0.5, 'unit': 'ng/mL',
            'description': 'Peak reproductive-phase estrogen (luteal / not used by default)'
        },
        'clearance_rate': {
            'default': 0.001, 'unit': '/min',
            'description': 'First-order clearance; half-life = ln2/0.001 ≈ 693 min (≈11.5 hr)'
        },
        'time_constant': {
            'default': 480.0, 'unit': 'min',
            'description': 'Equilibration time constant (used to modulate response speed)'
        },
    }
 
    def __init__(self, basal_estrogen=0.2, reproductive_estrogen=0.5,
                 clearance_rate=0.001, time_constant=480.0):
        super().__init__("estrogen_bone_protection", TimeScale.MINUTES)
        self.basal_estrogen      = basal_estrogen
        self.reproductive_estrogen = reproductive_estrogen
        self.clearance_rate      = clearance_rate
        self.time_constant       = time_constant    # min (not used in base ODE; available for extension)
 
    def step(self, state, dt):
        estrogen = state.get_signal('blood', 'estrogen')
 
        # First-order approach to basal set-point
        dE_dt = self.clearance_rate * (self.basal_estrogen - estrogen)   # ng/mL/min
        state.update_signal('blood', 'estrogen', dE_dt * (dt / 60.0))
 
 
# ===========================================================================
# STUB 6 ─ ThyroidHormoneEffects_ODE_GO0046850
# ===========================================================================
class ThyroidHormoneEffects_ODE_GO0046850(ProcessModel):
    """
    T3 (triiodothyronine) dynamics ODE.  The bone effect is exerted through
    blood.t3, which OsteoclastBoneResorption_GO0046850 reads to amplify resorption.
 
    Mechanism: T3 up-regulates RANK expression on osteoclast precursors and increases
    osteoclast sensitivity to RANKL, increasing bone resorption rate.
 
    PDF rows encoded:
    ┌──────────────────────────────────────────────────────────────────────────┐
    │ skeletal  │ t3 increases bone degradation (osteoclast activation)        │
    └──────────────────────────────────────────────────────────────────────────┘
 
    ODE:
        d[T3]/dt = (basal_t3 − [T3]) / time_constant
        (first-order equilibration; production balances clearance at set-point)
    """
 
    inputs  = {}
    outputs = {'t3': ('blood', 't3')}
 
    parameters = {
        'basal_t3': {
            'default': 1.0, 'unit': 'ng/mL',
            'description': 'T3 homeostatic set-point (normal 0.8–2.0 ng/mL)'
        },
        'time_constant': {
            'default': 600.0, 'unit': 'min',
            'description': 'T3 equilibration time (~10 hr; physiological T3 half-life 1–3 days)'
        },
        'clearance': {
            'default': 0.005, 'unit': '/min',
            'description': 'T3 clearance rate constant (not used in base ODE; for extension)'
        },
    }
 
    def __init__(self, basal_t3=1.0, time_constant=600.0, clearance=0.005):
        super().__init__("thyroid_hormone_effects", TimeScale.MINUTES)
        self.basal_t3      = basal_t3
        self.time_constant = time_constant
        self.clearance     = clearance
 
    def step(self, state, dt):
        t3 = state.get_signal('blood', 't3')
 
        dT3_dt = (self.basal_t3 - t3) / self.time_constant     # ng/mL/min
        state.update_signal('blood', 't3', dT3_dt * (dt / 60.0))

class OPGSecretion_GO0046850(ProcessModel):
    """
    Osteoprotegerin (OPG) secretion by osteoblasts / osteocytes.
 
    OPG is a decoy receptor that sequesters RANKL, reducing RANK signalling
    on osteoclast precursors. PTH suppresses OPG → more free RANKL → more
    osteoclastogenesis. Osteoblast activity (proxied by osteocalcin) drives
    OPG production.
 
    PDF rows encoded:
    - osteocyte / osteoblast does osteoprotegrin secretion
    - PTH decreases osteoprotegrin from osteocytes
    - osteoprotegrin decreases RANKL concentration (decoy receptor)
    - osteoprotegrin decreases bone degradation
    """
    inputs  = {
        'pth':        ('blood', 'parathyroid_hormone'),
        'osteocalcin':('blood', 'osteocalcin'),
    }
    outputs = {'opg': ('blood', 'opg')}
    parameters = {
        'basal_opg':   {'default': 1.0,  'unit': 'relative'},
        'K_pth':       {'default': 30.0, 'unit': 'pg/mL'},
        'clearance':   {'default': 0.002,'unit': '/min'},
    }
 
    def __init__(self, basal_opg=1.0, K_pth=30.0, clearance=0.002):
        super().__init__("opg_secretion", TimeScale.HOURS)
        self.basal_opg = basal_opg
        self.K_pth     = K_pth
        self.clearance = clearance
 
    def step(self, state, dt):
        pth        = state.get_signal('blood', 'parathyroid_hormone')
        osteocalcin= state.get_signal('blood', 'osteocalcin')
        opg        = state.get_signal('blood', 'opg')
 
        oc_factor   = osteocalcin / 20.0                          # normalised to basal
        pth_sup     = 1.0 / (1.0 + pth / self.K_pth)             # PTH suppresses OPG
        target      = self.basal_opg * oc_factor * pth_sup
        dopg        = self.clearance * (target - opg) * (dt / 60.0)
        state.update_signal('blood', 'opg', dopg)

class DietaryPhosphateModifiers_GO0006817(ProcessModel):
    """
    Gut-lumen phosphate sequestration: phytate chelation and Ca–P co-precipitation.
 
    1. Phytate (phytic acid in grains/legumes) chelates PO₄³⁻ → insoluble
       phytate-mineral complexes → fecal loss; reduces bioavailable phosphate.
 
    2. Ca–P co-precipitation: dietary Ca²⁺ × luminal PO₄³⁻ → CaHPO₄ / Ca₃(PO₄)₂
       precipitate in ileum/colon; both ions lost in feces proportionally.
 
    Net: removes a fraction of the dietary phosphate that would otherwise
    be absorbed by IntestinalPhosphateAbsorption_GO0006817.
 
    PDF rows encoded:
    - phytic acid increases PO₄³⁻ excretion to feces
    - Ca–P chelation in gut lumen increases precipitation and excretion to feces
    """
    inputs  = {
        'blood_calcium': ('blood', 'calcium'),
    }
    outputs = {'blood_phosphate': ('blood', 'phosphate')}
    parameters = {
        'dietary_phosphate':    {'default': 1000.0, 'unit': 'mg/day'},
        'phytate_loss_fraction':{'default': 0.12,   'unit': 'fraction',
                                 'description': '~12% of dietary PO4 bound by phytate in mixed diet'},
        'cap_coupling':         {'default': 0.003,  'unit': 'mg PO4 lost / (mg/dL Ca × mg/day phos)',
                                 'description': 'Ca–P chelation coupling constant'},
        'blood_volume_dL':      {'default': 50.0,   'unit': 'dL'},
    }
 
    def __init__(self, dietary_phosphate=1000.0, phytate_loss_fraction=0.12,
                 cap_coupling=0.003, blood_volume_dL=50.0):
        super().__init__("dietary_phosphate_modifiers", TimeScale.HOURS)
        self.dietary_phosphate     = dietary_phosphate
        self.phytate_loss_fraction = phytate_loss_fraction
        self.cap_coupling          = cap_coupling
        self.blood_volume_dL       = blood_volume_dL
 
    def step(self, state, dt):
        ca = state.get_signal('blood', 'calcium')
 
        # Phytate: constant fractional loss of dietary PO4 (independent of Ca)
        phytate_loss = self.dietary_phosphate * self.phytate_loss_fraction
 
        # Ca–P chelation: proportional to dietary Ca intake × dietary PO4
        # proxy: blood calcium as indicator of dietary calcium load
        cap_loss = self.cap_coupling * ca * self.dietary_phosphate
 
        total_loss = (phytate_loss + cap_loss) * (dt / 86400.0)   # mg per step
        state.update_signal('blood', 'phosphate', -total_loss / self.blood_volume_dL)
 
 
class IntestinalCalciumBarrier_GO0055074(ProcessModel):
    """
    CaSR-mediated tight junction control of paracellular Ca²⁺ flux.
 
    Luminal Ca²⁺ activates CaSR (calcium-sensing receptor) on enterocytes →
    tightens tight junctions → reduces paracellular back-flux of Ca²⁺ into the
    lumen → net increase in fractional absorption at higher dietary intake.
 
    Modelled as a blood-calcium-dependent positive correction to blood Ca²⁺
    (representing reduced paracellular back-loss). Effect is small; meaningful
    mainly when dietary Ca is elevated or calcitriol is high.
 
    PDF rows encoded:
    - Ca²⁺ influx (TRPV6) → CaSR activation → tightens tight junctions
    - tight junction permeability mediates passive Ca²⁺ absorption
    """
    inputs  = {'blood_calcium': ('blood', 'calcium')}
    outputs = {'blood_calcium': ('blood', 'calcium')}
    parameters = {
        'dietary_calcium':   {'default': 1000.0, 'unit': 'mg/day'},
        'max_barrier_effect':{'default': 0.04,   'unit': 'fraction of dietary Ca',
                              'description': 'Max extra fractional Ca retention from tight junctions'},
        'K_calcium':         {'default': 9.5,    'unit': 'mg/dL',
                              'description': 'Blood Ca for half-max CaSR activation'},
        'blood_volume_dL':   {'default': 50.0,   'unit': 'dL'},
    }
 
    def __init__(self, dietary_calcium=1000.0, max_barrier_effect=0.04,
                 K_calcium=9.5, blood_volume_dL=50.0):
        super().__init__("intestinal_calcium_barrier", TimeScale.HOURS)
        self.dietary_calcium    = dietary_calcium
        self.max_barrier_effect = max_barrier_effect
        self.K_calcium          = K_calcium
        self.blood_volume_dL    = blood_volume_dL
 
    def step(self, state, dt):
        ca = state.get_signal('blood', 'calcium')
        dietary_ca = state.get_organism_state('dietary_calcium', self.dietary_calcium)
 
        # CaSR activation by blood (proxy for luminal) Ca²⁺
        casr = ca / (self.K_calcium + ca)
 
        # Extra Ca retained due to tighter junctions
        extra = dietary_ca * self.max_barrier_effect * casr * (dt / 86400.0)
        state.update_signal('blood', 'calcium', extra / self.blood_volume_dL)
 