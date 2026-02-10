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