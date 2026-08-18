import numpy as np
from scipy.integrate import solve_ivp
from models.base import ProcessModel, TimeScale


class SomatostatinSecretion(ProcessModel):
    """
    Pancreatic delta-cell somatostatin secretion (SST-14)

    Row 22 (base). Glucose-driven Hill. K is set above the beta-cell K
    (110 vs 90) because delta cells ramp later on the islet glucose curve.

    Steady state = secretion / clearance, with the 2 min half-life in
    HormoneDegradation: ~11.7 pg/mL at glucose 90, ~17.9 at glucose 180.
    Table target [0, 30] pg/mL.
    """

    inputs = {
        'glucose': ('blood', 'glucose')
    }
    outputs = {
        'blood_somatostatin': ('blood', 'somatostatin')
    }

    parameters = {
        'basal_secretion': {
            'default': 1.4, 'unit': 'pg/mL/min', 'range': (0.5, 3.0),
            'description': 'Fasting delta-cell secretion rate'
        },
        'max_secretion': {
            'default': 8.0, 'unit': 'pg/mL/min', 'range': (4.0, 15.0),
            'description': 'Maximum glucose-stimulated secretion rate'
        },
        'glucose_sensitivity': {
            'default': 1.0, 'unit': 'dimensionless', 'range': (0.3, 2.0),
            'description': 'Delta-cell responsiveness to glucose'
        }
    }

    def __init__(self, basal_secretion=1.4, max_secretion=8.0, glucose_sensitivity=1.0):
        super().__init__("somatostatin_secretion", TimeScale.MINUTES)
        self.basal_secretion = basal_secretion
        self.max_secretion = max_secretion
        self.glucose_sensitivity = glucose_sensitivity

    def step(self, state, dt):
        glucose = state.get_signal('blood', 'glucose')
        if glucose is None or glucose <= 0:
            return

        K = 110.0   # mg/dL, half-maximal glucose
        n = 2.0     # Hill coefficient

        stimulated = (self.max_secretion - self.basal_secretion) * \
                     (glucose**n) / (K**n + glucose**n)
        secretion_rate = self.basal_secretion + stimulated * self.glucose_sensitivity

        state.update_signal('blood', 'somatostatin', secretion_rate * (dt/60.0))


class AmylinSecretion(ProcessModel):
    """
    Beta-cell amylin (IAPP) secretion

    Rows 28 (base) and 20 (insulin increases amylin). Amylin is co-packaged
    with insulin in beta-cell granules, so secretion is driven by blood
    insulin rather than glucose directly.

    Saturating (not linear) in insulin, representing granule depletion. This
    also keeps amylin bounded while blood insulin is running high.

    Steady state with the 13 min half-life in HormoneDegradation:
    ~6.0 pmol/L at insulin 5, ~27 at insulin 100. Table target [4.5, 9.5]
    fasting, 25-30 postprandial.

    NOTE on units: insulin here is µU/mL, amylin is pmol/L. insulin_coupling
    absorbs the conversion - do not treat these as the same scale.
    """

    inputs = {
        'insulin': ('blood', 'insulin')
    }
    outputs = {
        'blood_amylin': ('blood', 'amylin')
    }

    parameters = {
        'basal_secretion': {
            'default': 0.163, 'unit': 'pmol/L/min', 'range': (0.05, 0.5),
            'description': 'Insulin-independent amylin release'
        },
        'insulin_coupling': {
            'default': 2.044, 'unit': 'pmol/L/min', 'range': (1.0, 4.0),
            'description': 'Maximum insulin-driven secretion above basal'
        },
        'insulin_k': {
            'default': 60.0, 'unit': 'µU/mL', 'range': (20.0, 150.0),
            'description': 'Half-maximal insulin for co-secretion; sets where amylin saturates'
        }
    }

    def __init__(self, basal_secretion=0.163, insulin_coupling=2.044, insulin_k=60.0):
        super().__init__("amylin_secretion", TimeScale.MINUTES)
        self.basal_secretion = basal_secretion
        self.insulin_coupling = insulin_coupling
        self.insulin_k = insulin_k

    def step(self, state, dt):
        insulin = state.get_signal('blood', 'insulin')
        if insulin is None or insulin < 0:
            return

        secretion_rate = self.basal_secretion + \
                         self.insulin_coupling * insulin / (self.insulin_k + insulin)

        state.update_signal('blood', 'amylin', secretion_rate * (dt/60.0))


class InsulinSecretion(ProcessModel):
    """
    Beta-cell insulin secretion

    Rows: 13 base, 14 glucose↑, 16 fed↑, 15 somatostatin↓, 18 PP↓, 73 glp-1↑
    Missing: 19 (C-peptide)
    """

    inputs = {
        'glucose': ('blood', 'glucose'),
        'somatostatin': ('blood', 'somatostatin'),
        'pancreatic_polypeptide': ('blood', 'pancreatic_polypeptide'),
        'glp1': ('blood', 'glp1'),
        'fed_status': ('organism', 'fed_status')
    }
    outputs = {
        'blood_insulin': ('blood', 'insulin')
    }

    parameters = {
        'glucose_sensitivity': {
            'default': 1.0, 'unit': 'dimensionless', 'range': (0.3, 2.0),
            'description': 'Beta-cell responsiveness; 1.0=normal, <0.5=insulin resistant, >1.5=highly sensitive'
        },
        'basal_insulin': {
            'default': 0.85, 'unit': 'µU/mL', 'range': (0.2, 3.0),
            'description': 'TARGET CONCENTRATION at zero glucose, not a rate'
        },
        'max_insulin': {
            'default': 139.7, 'unit': 'µU/mL', 'range': (50.0, 300.0),
            'description': 'TARGET CONCENTRATION increment at saturating glucose, not a rate'
        },
        'clearance_per_min': {
            'default': 0.138629, 'unit': '/min', 'range': (0.01, 1.0),
            'description': 'MUST match insulin_half_life in HormoneDegradation: ln(2)/5'
        },
        'glucose_K': {
            'default': 155.0, 'unit': 'mg/dL', 'range': (100.0, 250.0),
            'description': 'Half-maximal glucose; above fasting on purpose'
        },
        'glucose_n': {
            'default': 6.0, 'unit': 'dimensionless', 'range': (1.5, 8.0),
            'description': 'Hill coefficient; GSIS is strongly cooperative'
        },
        'fed_multiplier': {
            'default': 1.15, 'unit': 'dimensionless', 'range': (1.0, 2.0),
            'description': 'Row 16. Modest because the glucose Hill term already carries most of the feeding signal; this is the glucose-independent part (incretin, vagal). Partly standing in for row 73 until GLP-1 exists.'
        },
        'fasted_multiplier': {
            'default': 0.85, 'unit': 'dimensionless', 'range': (0.4, 1.0),
            'description': 'Row 16, paired with fed_multiplier'
        },
        'sst_ref': {
            'default': 12.0, 'unit': 'pg/mL', 'range': (0.0, 30.0),
            'description': 'Row 15. Somatostatin level at which inhibition = 1.0'
        },
        'sst_k': {
            'default': 35.0, 'unit': 'pg/mL', 'range': (15.0, 80.0),
            'description': 'Row 15. Larger than the glucagon sst_k (20) because beta cells (SSTR5) are less somatostatin-sensitive than alpha cells (SSTR2)'
        },
        'pp_ref': {
            'default': 80.0, 'unit': 'pg/mL', 'range': (0.0, 150.0),
            'description': 'Row 18. PP level at which inhibition = 1.0'
        },
        'pp_k': {
            'default': 500.0, 'unit': 'pg/mL', 'range': (150.0, 1000.0),
            'description': 'Row 18. Half-maximal PP inhibition above pp_ref'
        },
        'glp1_ref': {
            'default': 5.0, 'unit': 'pmol/L', 'range': (0.0, 20.0),
            'description': 'Row 73. GLP-1 level at which amplification = 1.0 (no effect)'
        },
        'glp1_k': {
            'default': 30.0, 'unit': 'pmol/L', 'range': (10.0, 80.0),
            'description': 'Row 73. Half-maximal incretin amplification above glp1_ref'
        },
        'glp1_amp': {
            'default': 1.0, 'unit': 'dimensionless', 'range': (0.0, 2.0),
            'description': 'Row 73. Maximum fractional boost; 1.0 = up to a doubling of secretion at saturating GLP-1'
        }
    }

    def __init__(self, glucose_sensitivity=1.0, basal_insulin=0.85, max_insulin=139.7,
                 clearance_per_min=0.138629, glucose_K=155.0, glucose_n=6.0,
                 fed_multiplier=1.15, fasted_multiplier=0.85,
                 sst_ref=12.0, sst_k=35.0, pp_ref=80.0, pp_k=500.0,
                 glp1_ref=5.0, glp1_k=30.0, glp1_amp=1.0):
        super().__init__("insulin_secretion", TimeScale.MINUTES)
        self.glucose_sensitivity = glucose_sensitivity
        self.basal_insulin = basal_insulin
        self.max_insulin = max_insulin
        self.clearance_per_min = clearance_per_min
        self.glucose_K = glucose_K
        self.glucose_n = glucose_n
        self.fed_multiplier = fed_multiplier
        self.fasted_multiplier = fasted_multiplier
        self.sst_ref = sst_ref
        self.sst_k = sst_k
        self.pp_ref = pp_ref
        self.pp_k = pp_k
        self.glp1_ref = glp1_ref
        self.glp1_k = glp1_k
        self.glp1_amp = glp1_amp

    def step(self, state, dt):
        glucose = state.get_signal('blood', 'glucose')

        K = self.glucose_K
        n = self.glucose_n

        stimulated = self.max_insulin * (glucose**n) / (K**n + glucose**n)   # row 14
        target = self.basal_insulin + stimulated * self.glucose_sensitivity

        # row 16
        fed_status = state.get_organism_state('fed_status', 'fasted')
        target *= self.fed_multiplier if fed_status == 'fed' else self.fasted_multiplier

        # row 15
        sst = state.get_signal('blood', 'somatostatin')
        if sst is not None:
            target *= 1.0 / (1.0 + max(0.0, sst - self.sst_ref) / self.sst_k)

        # row 18
        pp = state.get_signal('blood', 'pancreatic_polypeptide')
        if pp is not None:
            target *= 1.0 / (1.0 + max(0.0, pp - self.pp_ref) / self.pp_k)

        # row 73
        glp1 = state.get_signal('blood', 'glp1')
        if glp1 is not None:
            excess = max(0.0, glp1 - self.glp1_ref)
            target *= 1.0 + self.glp1_amp * excess / (self.glp1_k + excess)

        secretion_rate = target * self.clearance_per_min
        state.update_signal('blood', 'insulin', secretion_rate * (dt / 60.0))


class BetaCellOscillator(ProcessModel):
    """Beta-cell insulin secretion WITH intrinsic pulsatility.

    Same steady-state dose-response as InsulinSecretion (glucose -> insulin LEVEL via
    the Hill term), but the secretion also carries an internal 2-state relaxation
    oscillator (FitzHugh-Nagumo form) standing in for the intracellular Ca2+/metabolic
    cycle that drives pulsatile secretion. Because the oscillator has its own state
    (self._a, self._r) it keeps cycling even at CONSTANT glucose -- the isolated-islet
    result a memoryless Hill map cannot reproduce.

    Design:
      tonic level  = basal + s * max_insulin * Hill_155(glucose)      (unchanged dose-response)
      oscillator   = tau_a a' = a - a^3/3 - r + drive(glucose)
                     tau_r r' = a - gamma r
      secretion    = tonic * (1 + pulse_frac * (max(a,0) - 0.5))       (pulses ride on the level)
    Mean insulin therefore tracks glucose exactly as before; the oscillator only adds
    the ~5-15 min rhythm on top, with pulse amplitude scaled by the glucose-driven level.
    """

    inputs = {
        'glucose': ('blood', 'glucose'),
        'somatostatin': ('blood', 'somatostatin'),
        'pancreatic_polypeptide': ('blood', 'pancreatic_polypeptide'),
        'glp1': ('blood', 'glp1'),
        'fed_status': ('organism', 'fed_status')
    }
    outputs = {
        'blood_insulin': ('blood', 'insulin')
    }

    parameters = {
        # --- dose-response (identical meaning to InsulinSecretion) ---
        'glucose_sensitivity': {'default': 1.0, 'unit': 'dimensionless', 'range': (0.3, 2.0),
                                'description': 'Beta-cell responsiveness'},
        'basal_insulin': {'default': 0.85, 'unit': 'uU/mL', 'range': (0.2, 3.0),
                          'description': 'Target concentration at zero glucose'},
        'max_insulin': {'default': 139.7, 'unit': 'uU/mL', 'range': (50.0, 300.0),
                       'description': 'Target increment at saturating glucose'},
        'clearance_per_min': {'default': 0.138629, 'unit': '/min', 'range': (0.01, 1.0),
                             'description': 'MUST match insulin_half_life in HormoneDegradation: ln(2)/5'},
        'glucose_K': {'default': 155.0, 'unit': 'mg/dL', 'range': (100.0, 250.0),
                     'description': 'Half-maximal glucose of the tonic dose-response'},
        'glucose_n': {'default': 6.0, 'unit': 'dimensionless', 'range': (1.5, 8.0),
                     'description': 'Hill coefficient of the tonic dose-response'},
        'fed_multiplier': {'default': 1.15, 'unit': 'dimensionless', 'range': (1.0, 2.0),
                          'description': 'Row 16 fed scaling'},
        'fasted_multiplier': {'default': 0.85, 'unit': 'dimensionless', 'range': (0.4, 1.0),
                             'description': 'Row 16 fasted scaling'},
        'sst_ref': {'default': 12.0, 'unit': 'pg/mL', 'range': (0.0, 30.0), 'description': 'Somatostatin ref'},
        'sst_k': {'default': 50.0, 'unit': 'pg/mL', 'range': (1.0, 200.0), 'description': 'Somatostatin half-max'},
        'pp_ref': {'default': 400.0, 'unit': 'pg/mL', 'range': (0.0, 1000.0), 'description': 'PP ref'},
        'pp_k': {'default': 500.0, 'unit': 'pg/mL', 'range': (1.0, 2000.0), 'description': 'PP half-max'},
        'glp1_ref': {'default': 5.0, 'unit': 'pmol/L', 'range': (0.0, 20.0), 'description': 'GLP-1 ref'},
        'glp1_k': {'default': 40.0, 'unit': 'pmol/L', 'range': (15.0, 100.0), 'description': 'GLP-1 half-max'},
        'glp1_amp': {'default': 0.5, 'unit': 'dimensionless', 'range': (0.0, 2.0), 'description': 'GLP-1 potentiation'},
        # --- intrinsic oscillator (the new mechanism) ---
        'osc_tau_a': {'default': 0.3, 'unit': 'min', 'range': (0.1, 2.0),
                     'description': 'Fast (activity) time constant of the beta-cell oscillator'},
        'osc_tau_r': {'default': 3.5, 'unit': 'min', 'range': (1.0, 20.0),
                     'description': 'Slow (recovery) time constant; sets pulse period (~10-12 min here)'},
        'osc_gamma': {'default': 0.8, 'unit': 'dimensionless', 'range': (0.2, 2.0),
                     'description': 'Recovery coupling of the oscillator'},
        'osc_beta0': {'default': -0.2, 'unit': 'dimensionless', 'range': (-1.0, 1.0),
                     'description': 'Baseline oscillator drive (bias)'},
        'osc_beta1': {'default': 0.9, 'unit': 'dimensionless', 'range': (0.0, 2.0),
                     'description': 'Glucose gain on oscillator drive'},
        'osc_K': {'default': 110.0, 'unit': 'mg/dL', 'range': (80.0, 200.0),
                 'description': 'Half-max glucose for the oscillator drive'},
        'osc_n': {'default': 4.0, 'unit': 'dimensionless', 'range': (1.0, 8.0),
                 'description': 'Hill coefficient for the oscillator drive'},
        'pulse_frac': {'default': 0.35, 'unit': 'dimensionless', 'range': (0.0, 1.0),
                      'description': 'Pulse amplitude as a fraction of the tonic level (0 = no pulsatility)'},
    }

    def __init__(self, glucose_sensitivity=1.0, basal_insulin=0.85, max_insulin=139.7,
                 clearance_per_min=0.138629, glucose_K=155.0, glucose_n=6.0,
                 fed_multiplier=1.15, fasted_multiplier=0.85,
                 sst_ref=12.0, sst_k=50.0, pp_ref=400.0, pp_k=500.0,
                 glp1_ref=5.0, glp1_k=40.0, glp1_amp=0.5,
                 osc_tau_a=0.3, osc_tau_r=3.5, osc_gamma=0.8,
                 osc_beta0=-0.2, osc_beta1=0.9, osc_K=110.0, osc_n=4.0, pulse_frac=0.35):
        super().__init__("insulin_secretion", TimeScale.MINUTES)
        self.glucose_sensitivity = glucose_sensitivity
        self.basal_insulin = basal_insulin
        self.max_insulin = max_insulin
        self.clearance_per_min = clearance_per_min
        self.glucose_K = glucose_K
        self.glucose_n = glucose_n
        self.fed_multiplier = fed_multiplier
        self.fasted_multiplier = fasted_multiplier
        self.sst_ref = sst_ref; self.sst_k = sst_k
        self.pp_ref = pp_ref; self.pp_k = pp_k
        self.glp1_ref = glp1_ref; self.glp1_k = glp1_k; self.glp1_amp = glp1_amp
        self.osc_tau_a = osc_tau_a; self.osc_tau_r = osc_tau_r; self.osc_gamma = osc_gamma
        self.osc_beta0 = osc_beta0; self.osc_beta1 = osc_beta1
        self.osc_K = osc_K; self.osc_n = osc_n; self.pulse_frac = pulse_frac
        self._a = 0.0   # oscillator activity  (internal state = the "memory")
        self._r = 0.0   # oscillator recovery

    def step(self, state, dt):
        glucose = state.get_signal('blood', 'glucose')

        # --- tonic dose-response (the LEVEL) : identical to InsulinSecretion ---
        K, n = self.glucose_K, self.glucose_n
        stimulated = self.max_insulin * (glucose**n) / (K**n + glucose**n)   # row 14
        tonic = self.basal_insulin + stimulated * self.glucose_sensitivity

        # --- advance the intrinsic oscillator (sub-stepped so a ~10 min pulse is resolved) ---
        m = dt / 60.0
        drive = self.osc_beta0 + self.osc_beta1 * (glucose**self.osc_n) / (self.osc_K**self.osc_n + glucose**self.osc_n)
        nsub = max(1, int(np.ceil(m / 0.1)))     # <=0.1 min integration steps
        h = m / nsub
        for _ in range(nsub):
            da = (self._a - self._a**3/3.0 - self._r + drive) / self.osc_tau_a
            dr = (self._a - self.osc_gamma * self._r) / self.osc_tau_r
            self._a += h * da
            self._r += h * dr

        # --- pulses ride on the tonic level ---
        target = tonic * (1.0 + self.pulse_frac * (max(self._a, 0.0) - 0.5))

        # --- same modulators as InsulinSecretion ---
        fed_status = state.get_organism_state('fed_status', 'fasted')
        target *= self.fed_multiplier if fed_status == 'fed' else self.fasted_multiplier
        sst = state.get_signal('blood', 'somatostatin')
        if sst is not None:
            target *= 1.0 / (1.0 + max(0.0, sst - self.sst_ref) / self.sst_k)
        pp = state.get_signal('blood', 'pancreatic_polypeptide')
        if pp is not None:
            target *= 1.0 / (1.0 + max(0.0, pp - self.pp_ref) / self.pp_k)
        glp1 = state.get_signal('blood', 'glp1')
        if glp1 is not None:
            excess = max(0.0, glp1 - self.glp1_ref)
            target *= 1.0 + self.glp1_amp * excess / (self.glp1_k + excess)

        target = max(0.0, target)
        secretion_rate = target * self.clearance_per_min
        state.update_signal('blood', 'insulin', secretion_rate * (dt / 60.0))


class GlucagonSecretion(ProcessModel):
    """
    Alpha-cell glucagon secretion

    Rows: 5 base, 7 glucose↓, 6 insulin↓, 9/17 fed↓ fasted↑,
          10 PP↓, 8 somatostatin↓, 11/30 amylin↓, 74 glp-1↓
    Missing: none - all 10 glucagon rows implemented
    """

    inputs = {
        'glucose': ('blood', 'glucose'),
        'insulin': ('blood', 'insulin'),
        'pancreatic_polypeptide': ('blood', 'pancreatic_polypeptide'),
        'somatostatin': ('blood', 'somatostatin'),
        'amylin': ('blood', 'amylin'),
        'glp1': ('blood', 'glp1'),
        'fed_status': ('organism', 'fed_status')
    }
    outputs = {
        'blood_glucagon': ('blood', 'glucagon')
    }

    parameters = {
        'basal_glucagon': {
            'default': 135.7, 'unit': 'pg/mL', 'range': (60.0, 300.0),
            'description': 'TARGET CONCENTRATION scale, not a rate. Same units fix as InsulinSecretion: the old parameter was pg/mL/min applied directly, giving ~519 pg/mL against the 6 min clearance half-life.'
        },
        'clearance_per_min': {
            'default': 0.115525, 'unit': '/min', 'range': (0.01, 1.0),
            'description': 'MUST match glucagon_half_life in HormoneDegradation: ln(2)/6'
        },
        'fasted_multiplier': {
            'default': 1.15, 'unit': 'dimensionless', 'range': (1.0, 2.0),
            'description': 'Rows 9/17'
        },
        'fed_multiplier': {
            'default': 0.85, 'unit': 'dimensionless', 'range': (0.4, 1.0),
            'description': 'Rows 9/17'
        },
        'pp_ref': {
            'default': 80.0, 'unit': 'pg/mL', 'range': (0.0, 150.0),
            'description': 'Row 10. PP level at which inhibition = 1.0'
        },
        'pp_k': {
            'default': 500.0, 'unit': 'pg/mL', 'range': (150.0, 1000.0),
            'description': 'Row 10. Half-maximal PP inhibition above pp_ref'
        },
        'sst_ref': {
            'default': 12.0, 'unit': 'pg/mL', 'range': (0.0, 30.0),
            'description': 'Row 8. Somatostatin level at which inhibition = 1.0'
        },
        'sst_k': {
            'default': 20.0, 'unit': 'pg/mL', 'range': (10.0, 60.0),
            'description': 'Row 8. Half-maximal somatostatin inhibition above sst_ref'
        },
        'amylin_ref': {
            'default': 6.0, 'unit': 'pmol/L', 'range': (0.0, 15.0),
            'description': 'Rows 11/30. Amylin level at which inhibition = 1.0'
        },
        'amylin_k': {
            'default': 50.0, 'unit': 'pmol/L', 'range': (20.0, 150.0),
            'description': 'Rows 11/30. Half-maximal amylin inhibition above amylin_ref'
        },
        'glp1_ref': {
            'default': 5.0, 'unit': 'pmol/L', 'range': (0.0, 20.0),
            'description': 'Row 74. GLP-1 level at which inhibition = 1.0'
        },
        'glp1_k': {
            'default': 40.0, 'unit': 'pmol/L', 'range': (15.0, 100.0),
            'description': 'Row 74. Half-maximal glucagon suppression above glp1_ref'
        },
        'insulin_k': {
            'default': 40.0, 'unit': 'uU/mL', 'range': (10.0, 100.0),
            'description': 'Row 6. Half-maximal insulin suppression; higher = gentler, more graded alpha-cell response'
        }
    }

    def __init__(self, basal_glucagon=155.0, clearance_per_min=0.115525,
                 fasted_multiplier=1.15,
                 fed_multiplier=0.85, pp_ref=80.0, pp_k=500.0,
                 sst_ref=12.0, sst_k=20.0, amylin_ref=6.0, amylin_k=50.0,
                 glp1_ref=5.0, glp1_k=40.0, insulin_k=40.0):
        super().__init__("glucagon_secretion", TimeScale.MINUTES)
        self.basal_glucagon = basal_glucagon
        self.clearance_per_min = clearance_per_min
        self.fasted_multiplier = fasted_multiplier
        self.fed_multiplier = fed_multiplier
        self.pp_ref = pp_ref
        self.pp_k = pp_k
        self.sst_ref = sst_ref
        self.sst_k = sst_k
        self.amylin_ref = amylin_ref
        self.amylin_k = amylin_k
        self.glp1_ref = glp1_ref
        self.glp1_k = glp1_k
        self.insulin_k = insulin_k

    def step(self, state, dt):
        glucose = state.get_signal('blood', 'glucose')
        insulin = state.get_signal('blood', 'insulin')

        glucose_factor = 70.0 / (glucose + 1.0)              # row 7
        insulin_inhibition = 1.0 / (1.0 + insulin / self.insulin_k)   # graded, no deadband
        target = self.basal_glucagon * glucose_factor * insulin_inhibition

        # rows 9 / 17
        fed_status = state.get_organism_state('fed_status', 'fasted')
        target *= self.fed_multiplier if fed_status == 'fed' else self.fasted_multiplier

        # row 10
        pp = state.get_signal('blood', 'pancreatic_polypeptide')
        if pp is not None:
            target *= 1.0 / (1.0 + max(0.0, pp - self.pp_ref) / self.pp_k)

        # row 8
        sst = state.get_signal('blood', 'somatostatin')
        if sst is not None:
            target *= 1.0 / (1.0 + max(0.0, sst - self.sst_ref) / self.sst_k)

        # rows 11 / 30
        amylin = state.get_signal('blood', 'amylin')
        if amylin is not None:
            target *= 1.0 / (1.0 + max(0.0, amylin - self.amylin_ref) / self.amylin_k)

        # row 74
        glp1 = state.get_signal('blood', 'glp1')
        if glp1 is not None:
            target *= 1.0 / (1.0 + max(0.0, glp1 - self.glp1_ref) / self.glp1_k)

        secretion_rate = target * self.clearance_per_min
        state.update_signal('blood', 'glucagon', secretion_rate * (dt / 60.0))



class RemoteInsulin(ProcessModel):
    """Plasma insulin -> delayed 'remote' (interstitial / receptor-level) insulin action.

    ONE biological mechanism: the transport + receptor-kinetics lag between plasma
    insulin and the insulin signal that actually acts on tissue. Modelled as an
    n-stage first-order chain (a linear, gamma-distributed delay) that approximates
    a pure time delay of ~ n_stages * action_td_min minutes.

    Both peripheral glucose uptake and hepatic glucose production read this single
    'blood.remote_insulin' signal instead of instantaneous plasma insulin. That shared
    delay around the negative-feedback loop is what lets the glucose-insulin system
    cross a Hopf bifurcation into self-sustained ultradian oscillation
    (Sturis/Tolic), rather than settling to a flat fixed point.
    """

    inputs = {
        'plasma_insulin': ('blood', 'insulin'),
    }
    outputs = {
        'remote_insulin': ('blood', 'remote_insulin'),
    }
    parameters = {
        'action_td_min': {
            'default': 12.0, 'unit': 'min', 'range': (3.0, 30.0),
            'description': 'Per-stage time constant of the insulin-action delay chain. '
                           'Total delay ~= n_stages * action_td_min. Longer delay (with enough '
                           'loop gain) pushes the glucose-insulin loop into sustained oscillation.'
        },
        'n_stages': {
            'default': 3, 'unit': 'count', 'range': (1, 5),
            'description': 'Number of first-order compartments in the delay chain. 3 reproduces '
                           'the classic ~100-120 min ultradian oscillation.'
        },
    }

    def __init__(self, action_td_min=12.0, n_stages=3):
        super().__init__("remote_insulin", TimeScale.MINUTES)
        self.action_td_min = action_td_min
        self.n_stages = int(n_stages)
        self._x = None  # lazily initialised to plasma insulin on first step

    def step(self, state, dt):
        plasma = state.get_signal('blood', 'insulin')
        if plasma is None:
            return
        # Start the chain at the current plasma level so the run begins at steady
        # state instead of ramping from zero.
        if self._x is None:
            self._x = [plasma] * self.n_stages

        a = 1.0 - np.exp(-(dt / 60.0) / self.action_td_min)
        prev = plasma
        for i in range(self.n_stages):
            self._x[i] += (prev - self._x[i]) * a
            prev = self._x[i]

        # remote_insulin is a computed state, not a flux -> set, don't accumulate.
        state.set_signal('blood', 'remote_insulin', self._x[-1])


class GlucoseUptake(ProcessModel):
    """
    Insulin-mediated glucose uptake by tissues

    Mechanism: Basal + insulin-stimulated GLUT4 transport
    Timescale: Minutes
    Integration: RK45 (via scipy solve_ivp), insulin held frozen over the step

    Rows 80 / 82: adiponectin raises and resistin lowers insulin sensitivity.
    These modulate the sensitivity term rather than the rate directly - the
    table verb is "sensivitity to X", not a secretion or uptake rate, so the
    adipokines change how strongly insulin acts rather than adding flux.
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
            'description': 'Baseline tissue insulin sensitivity before adipokine modulation; 1.0=normal, <0.5=resistant, >1.5=athlete'
        },
        'adiponectin_ref': {
            'default': 10.0, 'unit': 'µg/mL', 'range': (0.0, 30.0),
            'description': 'Row 80. Adiponectin level at which sensitivity is unmodified'
        },
        'adiponectin_k': {
            'default': 15.0, 'unit': 'µg/mL', 'range': (5.0, 50.0),
            'description': 'Row 80. Half-maximal sensitisation above adiponectin_ref'
        },
        'adiponectin_amp': {
            'default': 0.5, 'unit': 'dimensionless', 'range': (0.0, 1.5),
            'description': 'Row 80. Maximum fractional gain in sensitivity at saturating adiponectin'
        },
        'resistin_ref': {
            'default': 12.0, 'unit': 'ng/mL', 'range': (0.0, 30.0),
            'description': 'Row 82. Resistin level at which sensitivity is unmodified'
        },
        'resistin_k': {
            'default': 25.0, 'unit': 'ng/mL', 'range': (10.0, 80.0),
            'description': 'Row 82. Half-maximal desensitisation above resistin_ref'
        }
    }

    def __init__(self, target_entity='muscle_tissue', basal_rate=0.1, insulin_sensitivity=1.0,
                 adiponectin_ref=10.0, adiponectin_k=15.0, adiponectin_amp=0.5,
                 resistin_ref=12.0, resistin_k=25.0,
                 circadian_amp=0.0, circadian_acrophase_h=16.0,
                 sens_noise_sigma=0.0, noise_tau_min=90.0, noise_seed=0):
        super().__init__(f"glucose_uptake_{target_entity}", TimeScale.MINUTES)
        self.target_entity = target_entity
        self.basal_rate = basal_rate
        self.insulin_sensitivity = insulin_sensitivity
        self.adiponectin_ref = adiponectin_ref
        self.adiponectin_k = adiponectin_k
        self.adiponectin_amp = adiponectin_amp
        self.resistin_ref = resistin_ref
        self.resistin_k = resistin_k
        # --- within-individual variation (off by default) ---
        self.circadian_amp = circadian_amp              # fractional swing in insulin sensitivity over the day
        self.circadian_acrophase_h = circadian_acrophase_h  # hour of peak sensitivity; nadir ~12 h earlier (dawn)
        self.sens_noise_sigma = sens_noise_sigma        # stationary SD of correlated biological noise
        self.noise_tau_min = noise_tau_min              # correlation time of that noise (min)
        self._sens_noise = 0.0
        self._rng = np.random.default_rng(noise_seed + (hash(target_entity) % 9999))

        self.inputs = {
            'blood_glucose': ('blood', 'glucose'),
            'blood_insulin': ('blood', 'insulin'),
            'blood_adiponectin': ('blood', 'adiponectin'),
            'blood_resistin': ('blood', 'resistin')
        }
        self.outputs = {
            'blood_glucose': ('blood', 'glucose'),
            'target_glucose': (target_entity, 'glucose')
        }

    def effective_sensitivity(self, state):
        """Rows 80 / 82: adipokine modulation of insulin sensitivity."""
        sens = self.insulin_sensitivity

        adiponectin = state.get_signal('blood', 'adiponectin')
        if adiponectin is not None:
            excess = max(0.0, adiponectin - self.adiponectin_ref)
            sens *= 1.0 + self.adiponectin_amp * excess / (self.adiponectin_k + excess)

        resistin = state.get_signal('blood', 'resistin')
        if resistin is not None:
            sens *= 1.0 / (1.0 + max(0.0, resistin - self.resistin_ref) / self.resistin_k)

        # circadian modulation: sensitivity peaks at acrophase, dips ~12 h earlier (dawn phenomenon)
        if self.circadian_amp:
            hour = (state.time % 86400) / 3600.0
            sens *= 1.0 + self.circadian_amp * np.cos(2.0 * np.pi * (hour - self.circadian_acrophase_h) / 24.0)

        # temporally-correlated biological noise (OU), advanced in step()
        sens *= max(0.2, 1.0 + self._sens_noise)

        return sens

    def rate(self, glucose, insulin, sensitivity):
        """
        dG/dt in mg/dL/min — pure function, no state access, no mutation.
        insulin and sensitivity are frozen for this integration window.
        """
        insulin_factor = 1.0 + (12.0 * insulin / (60.0 + insulin)) * sensitivity
        return self.basal_rate * insulin_factor * (glucose / 90.0)

    def step(self, state, dt):
        G0 = state.get_signal('blood', 'glucose')
        # Insulin action is delayed: read the shared remote-insulin signal produced
        # by RemoteInsulin. Fall back to plasma insulin on the very first step before
        # that signal exists.
        I = state.get_signal('blood', 'remote_insulin')
        if not I:
            I = state.get_signal('blood', 'insulin') or 0.0
        dt_min = dt / 60.0

        if dt_min <= 0 or G0 <= 0:
            return

        if self.sens_noise_sigma > 0.0:
            theta = dt_min / self.noise_tau_min
            self._sens_noise += (-theta * self._sens_noise
                                 + self.sens_noise_sigma * np.sqrt(2.0 * theta) * self._rng.standard_normal())

        sens = self.effective_sensitivity(state)

        def dGdt(t, y):
            G = y[0]
            return [-self.rate(max(G, 0.0), I, sens)]

        sol = solve_ivp(
            dGdt,
            t_span=(0.0, dt_min),
            y0=[G0],
            method='RK45',
            rtol=1e-6,
            atol=1e-9
        )

        if not sol.success:
            amount = self.rate(G0, I, sens) * dt_min
        else:
            G_new = sol.y[0, -1]
            amount = G0 - G_new

        amount = max(0.0, min(amount, G0))

        state.update_signal('blood', 'glucose', -amount)
        state.update_signal(self.target_entity, 'glucose', amount)


class GlycogenSynthesis(ProcessModel):
    """
    Glycogen synthesis in liver and muscle
    Self-referential (per tissue): synthesis rate depends on local tissue
    glucose, which this process is depleting. insulin/fed_status are frozen
    inputs, checked once before integrating.
    """

    inputs = {
        'fed_status': ('fed_status', None, 'organism'),
        'liver_glucose': ('liver', 'glucose'),
        'blood_insulin': ('blood', 'insulin'),
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
        super().__init__("glycogen_synthesis", TimeScale.MINUTES)

    def rate(self, local_glucose, insulin, k):
        return k * local_glucose * (insulin / 10.0)

    def _integrate_tissue(self, state, entity, insulin, k, dt_hr, volume_dL):
            G0 = state.get_signal(entity, 'glucose')
            if G0 is None or G0 <= 0:
                return

            def dGdt(t, y):
                return [-self.rate(max(y[0], 0.0), insulin, k)]

            sol = solve_ivp(dGdt, (0.0, dt_hr), [G0], method='RK45', rtol=1e-6, atol=1e-9)
            amount = (G0 - sol.y[0, -1]) if sol.success else self.rate(G0, insulin, k) * dt_hr
            amount = max(0.0, min(amount, G0))

            state.update_signal(entity, 'glucose', -amount)
            state.update_signal(entity, 'glycogen', amount * volume_dL / 1000.0)

    def step(self, state, dt):
        fed_status = state.get_organism_state('fed_status', 'fasted')
        if fed_status != 'fed':
            return

        insulin = state.get_signal('blood', 'insulin')
        if insulin <= 8.0:
            return

        dt_hr = dt / 3600.0
        self._integrate_tissue(state, 'liver', insulin, k=1.5, dt_hr=dt_hr, volume_dL=15.0)
        self._integrate_tissue(state, 'muscle_tissue', insulin, k=0.02, dt_hr=dt_hr, volume_dL=250.0)

class GlycogenBreakdown(ProcessModel):
    """
    Glycogen breakdown (glycogenolysis), liver and muscle

    Liver rows 53-56, muscle rows 57-60.

    Liver and muscle differ in where the glucose goes. Hepatocytes have
    glucose-6-phosphatase and export free glucose to blood. Myocytes do NOT -
    the G6P is trapped and consumed locally, so muscle glycogen feeds
    muscle_tissue.glucose, never blood.glucose.
    """

    inputs = {
        'fed_status': ('organism', 'fed_status'),
        'glucagon': ('blood', 'glucagon'),
        'liver_glycogen': ('liver', 'glycogen'),
        'muscle_glycogen': ('muscle_tissue', 'glycogen')
    }
    outputs = {
        'liver_glycogen': ('liver', 'glycogen'),
        'blood_glucose': ('blood', 'glucose'),
        'muscle_glycogen': ('muscle_tissue', 'glycogen'),
        'muscle_glucose': ('muscle_tissue', 'glucose')
    }

    parameters = {
        'liver_rate': {
            'default': 0.5, 'unit': 'g/min', 'range': (0.1, 2.0),
            'description': 'Hepatic breakdown rate at glucagon = 80 pg/mL'
        },
        'muscle_rate': {
            'default': 0.15, 'unit': 'g/min', 'range': (0.0, 1.0),
            'description': 'Row 57. Muscle breakdown rate at glucagon = 80; lower than liver, resting muscle turns glycogen over slowly'
        },
        'glucagon_threshold': {
                    'default': 45.0, 'unit': 'pg/mL', 'range': (20.0, 120.0),
                    'description': 'Rows 56/60. Half-maximal glucagon for phosphorylase activation (Hill n=2), not a hard cutoff'
                },
        'muscle_glucagon_sensitivity': {
            'default': 1.0, 'unit': 'dimensionless', 'range': (0.0, 1.0),
            'description': 'Row 60 as written. NOTE: skeletal muscle expresses no glucagon receptor - the real drivers are epinephrine (beta2) and contraction. Set 0.0 to disable, and revisit when rows 62-64 add catecholamines.'
        },
        'muscle_volume_dL': {
            'default': 250.0, 'unit': 'dL', 'range': (100.0, 400.0),
            'description': 'Intracellular water of skeletal muscle, for the g -> mg/dL conversion'
        }
    }

    def __init__(self, liver_rate=0.5, muscle_rate=0.15, glucagon_threshold=45.0,
                 muscle_glucagon_sensitivity=1.0, muscle_volume_dL=250.0):
        super().__init__("glycogen_breakdown", TimeScale.MINUTES)
        self.liver_rate = liver_rate
        self.muscle_rate = muscle_rate
        self.glucagon_threshold = glucagon_threshold
        self.muscle_glucagon_sensitivity = muscle_glucagon_sensitivity
        self.muscle_volume_dL = muscle_volume_dL

    def step(self, state, dt):
        fed_status = state.get_organism_state('fed_status', 'fasted')
        glucagon = state.get_signal('blood', 'glucagon')

        # rows 54/55, 58/59
        if fed_status != 'fasted' or glucagon is None or glucagon <= 0:
                    return

        dt_min = dt / 60.0
        # smooth phosphorylase activation - a hard threshold here sat inside
        # glucagon's own fluctuation band and produced an 18 min relaxation oscillation
        activation = glucagon**2 / (self.glucagon_threshold**2 + glucagon**2)
        drive = (glucagon / 80.0) * activation

        # --- liver: rows 53-56, exports to blood (has G6Pase) ---
        liver_glycogen = state.get_signal('liver', 'glycogen')
        if liver_glycogen is not None and liver_glycogen > 0:
            grams = min(self.liver_rate * drive * dt_min, liver_glycogen)
            state.update_signal('liver', 'glycogen', -grams)
            # 1 g -> 1000 mg over 50 dL blood = 20 mg/dL
            state.update_signal('blood', 'glucose', grams * 20.0)

        # --- muscle: rows 57-60, stays local (no G6Pase) ---
        muscle_glycogen = state.get_signal('muscle_tissue', 'glycogen')
        if muscle_glycogen is not None and muscle_glycogen > 0:
            muscle_drive = 1.0 + (drive - 1.0) * self.muscle_glucagon_sensitivity
            grams = min(self.muscle_rate * muscle_drive * dt_min, muscle_glycogen)
            state.update_signal('muscle_tissue', 'glycogen', -grams)
            state.update_signal('muscle_tissue', 'glucose', grams * 1000.0 / self.muscle_volume_dL)

class HepaticGlucoseProduction(ProcessModel):
    """
    Hepatic glucose production (gluconeogenesis)

    Mechanism: Glucagon-stimulated, insulin-suppressed, glucose-autoregulated
    Timescale: Minutes

    Rows: 32 base, 33 glucagon↑, 34 insulin↓, 35 fed↓, 36 fasted↑,
          38 glucose↓, 37 cortisol↑
    Missing: none
    """

    inputs = {
        'glucagon': ('blood', 'glucagon'),
        'insulin': ('blood', 'insulin'),
        'glucose': ('blood', 'glucose'),
        'cortisol': ('blood', 'cortisol'),
        'fed_status': ('organism', 'fed_status')
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
        },
        'glucose_ref': {
            'default': 90.0,
            'unit': 'mg/dL',
            'range': (60.0, 120.0),
            'description': 'Row 38. Blood glucose at which autoregulation = 1.0'
        },
        'glucose_k': {
            'default': 60.0,
            'unit': 'mg/dL',
            'range': (30.0, 150.0),
            'description': 'Row 38. Half-maximal suppression above glucose_ref; 60 puts HGP at ~50% by 150 mg/dL'
        },
        'cortisol_ref': {
            'default': 10.0, 'unit': 'ug/dL', 'range': (0.0, 25.0),
            'description': 'Row 37. Cortisol level at which the gluconeogenic boost = 1.0'
        },
        'cortisol_k': {
            'default': 15.0, 'unit': 'ug/dL', 'range': (5.0, 50.0),
            'description': 'Row 37. Half-maximal cortisol stimulation above cortisol_ref'
        },
        'cortisol_amp': {
            'default': 0.8, 'unit': 'dimensionless', 'range': (0.0, 2.0),
            'description': 'Row 37. Maximum fractional boost to gluconeogenesis at saturating cortisol'
        }
    }

    def __init__(self, production_rate=2.0, glucose_ref=90.0, glucose_k=60.0,
                 cortisol_ref=10.0, cortisol_k=15.0, cortisol_amp=0.8):
        super().__init__("hepatic_glucose_production", TimeScale.MINUTES)
        self.production_rate = production_rate
        self.glucose_ref = glucose_ref
        self.glucose_k = glucose_k
        self.cortisol_ref = cortisol_ref
        self.cortisol_k = cortisol_k
        self.cortisol_amp = cortisol_amp

    def step(self, state, dt):
        glucagon = state.get_signal('blood', 'glucagon')
        # Delayed insulin action (same shared signal peripheral uptake uses), so both
        # arms of the loop carry the delay that drives the oscillation.
        insulin = state.get_signal('blood', 'remote_insulin')
        if not insulin:
            insulin = state.get_signal('blood', 'insulin')
        glucose = state.get_signal('blood', 'glucose')
        fed_status = state.get_organism_state('fed_status', 'fasted')

        glucagon_factor = glucagon / 60.0                       # row 33
        insulin_factor = 1.0 / (1.0 + insulin / 10.0)           # row 34
        fasting_boost = 1.5 if fed_status == 'fasted' else 0.5  # rows 35 / 36

        body_weight_kg = 70.0
        blood_volume_dL = 50.0  # 5 L

        # Calculate production in mg/min for whole body
        production_mg_per_min = (self.production_rate * glucagon_factor *
                                 insulin_factor * fasting_boost * body_weight_kg)

        # Row 38: hepatic autoregulation - high blood glucose suppresses output
        if glucose is not None:
            production_mg_per_min *= 1.0 / (1.0 + max(0.0, glucose - self.glucose_ref) / self.glucose_k)

        # Row 37: cortisol drives gluconeogenesis
        cortisol = state.get_signal('blood', 'cortisol')
        if cortisol is not None:
            excess = max(0.0, cortisol - self.cortisol_ref)
            production_mg_per_min *= 1.0 + self.cortisol_amp * excess / (self.cortisol_k + excess)

        production_concentration_per_min = production_mg_per_min / blood_volume_dL  # mg/dL/min

        # dt is in seconds for MINUTES timescale
        amount = production_concentration_per_min * (dt / 60.0)

        state.update_signal('blood', 'glucose', amount)

# Three new classes for models/endocrine.py - gut / incretin block
# Rows 29, 72, 74, 75 (+ infrastructure for the meal response)

import numpy as np
from models.base import ProcessModel, TimeScale


class GastricEmptying(ProcessModel):
    """
    Gastric emptying of carbohydrate into the small intestine

    Rows 29 / 75: glp-1 decreases export of gastric contents.
    Also slowed by amylin (row 28 QuantitativeNotes).

    ZERO-ORDER, not first-order. The duodenum meters caloric delivery at a
    roughly constant 2-4 kcal/min regardless of how much is in the stomach.
    First-order emptying (rate proportional to contents) trails a long
    exponential tail: a 70 g meal took 4.9 h to 90% empty against a real
    2-3 h, and the smeared delivery meant glucose appearance never exceeded
    disposal by enough to produce a postprandial peak.

    Michaelis-Menten rather than strict zero-order so the stomach tapers
    smoothly at the end instead of stopping at a corner. Emptying is
    effectively zero-order while contents >> half_saturation_g and
    first-order below it.

    Timescale: Minutes
    """

    inputs = {
        'gastric_contents': ('stomach', 'gastric_contents'),
        'glp1': ('blood', 'glp1'),
        'amylin': ('blood', 'amylin')
    }
    outputs = {
        'gastric_contents': ('stomach', 'gastric_contents'),
        'luminal_glucose': ('small_intestine', 'luminal_glucose')
    }

    parameters = {
        'max_emptying_rate': {
            'default': 1.25, 'unit': 'g/min', 'range': (0.3, 2.0),
            'description': 'Zero-order caloric metering rate. 1.0 g carbohydrate/min = 4 kcal/min, the top of the physiological 2-4 kcal/min band; 1.25 is slightly above it and is the value that reproduces a realistic postprandial peak given the rest of the model.'
        },
        'half_saturation_g': {
            'default': 30.0, 'unit': 'g', 'range': (1.0, 60.0),
            'description': 'Michaelis constant for the taper. Zero-order while contents >> this, first-order below it.'
        },
        'glp1_ref': {
            'default': 5.0, 'unit': 'pmol/L', 'range': (0.0, 20.0),
            'description': 'Rows 29/75. GLP-1 level at which the brake = 1.0'
        },
        'glp1_k': {
            'default': 35.0, 'unit': 'pmol/L', 'range': (10.0, 100.0),
            'description': 'Rows 29/75. Half-maximal slowing above glp1_ref'
        },
        'amylin_ref': {
            'default': 6.0, 'unit': 'pmol/L', 'range': (0.0, 15.0),
            'description': 'Amylin level at which the brake = 1.0'
        },
        'amylin_k': {
            'default': 40.0, 'unit': 'pmol/L', 'range': (15.0, 120.0),
            'description': 'Half-maximal slowing above amylin_ref'
        }
    }

    def __init__(self, max_emptying_rate=1.6, half_saturation_g=30.0,
                 glp1_ref=5.0, glp1_k=35.0, amylin_ref=6.0, amylin_k=40.0):
        super().__init__("gastric_emptying", TimeScale.MINUTES)
        self.max_emptying_rate = max_emptying_rate
        self.half_saturation_g = half_saturation_g
        self.glp1_ref = glp1_ref
        self.glp1_k = glp1_k
        self.amylin_ref = amylin_ref
        self.amylin_k = amylin_k

    def step(self, state, dt):
        contents = state.get_signal('stomach', 'gastric_contents')
        if contents is None or contents <= 0:
            return

        # zero-order caloric metering, Michaelis-Menten taper as the stomach empties
        rate = self.max_emptying_rate * contents / (self.half_saturation_g + contents)

        # rows 29 / 75
        glp1 = state.get_signal('blood', 'glp1')
        if glp1 is not None:
            rate *= 1.0 / (1.0 + max(0.0, glp1 - self.glp1_ref) / self.glp1_k)

        amylin = state.get_signal('blood', 'amylin')
        if amylin is not None:
            rate *= 1.0 / (1.0 + max(0.0, amylin - self.amylin_ref) / self.amylin_k)

        emptied = min(rate * (dt / 60.0), contents)

        state.update_signal('stomach', 'gastric_contents', -emptied)
        state.update_signal('small_intestine', 'luminal_glucose', emptied)

class IntestinalGlucoseAbsorption(ProcessModel):
    """
    SGLT1/GLUT2 glucose absorption from small intestine into blood

    Not a table row in the glucose sandbox - infrastructure that connects
    gastric emptying to blood glucose. First-order, near-complete.

    Timescale: Minutes
    """

    inputs = {
        'luminal_glucose': ('small_intestine', 'luminal_glucose')
    }
    outputs = {
        'luminal_glucose': ('small_intestine', 'luminal_glucose'),
        'blood_glucose': ('blood', 'glucose')
    }

    parameters = {
        'half_absorption_min': {
            'default': 20.0, 'unit': 'minutes', 'range': (5.0, 90.0),
            'description': 'Half-time for luminal glucose to cross into blood'
        },
        'blood_volume_dL': {
            'default': 50.0, 'unit': 'dL', 'range': (30.0, 70.0),
            'description': 'For the g -> mg/dL conversion; 1 g into 50 dL = 20 mg/dL'
        }
    }

    def __init__(self, half_absorption_min=20.0, blood_volume_dL=50.0):
        super().__init__("intestinal_glucose_absorption", TimeScale.MINUTES)
        self.half_absorption_min = half_absorption_min
        self.blood_volume_dL = blood_volume_dL

    def step(self, state, dt):
        lumen = state.get_signal('small_intestine', 'luminal_glucose')
        if lumen is None or lumen <= 0:
            return

        k = np.log(2) / self.half_absorption_min
        absorbed = min(lumen * k * (dt / 60.0), lumen)

        state.update_signal('small_intestine', 'luminal_glucose', -absorbed)
        state.update_signal('blood', 'glucose', absorbed * 1000.0 / self.blood_volume_dL)


class GLP1Secretion(ProcessModel):
    """
    Intestinal L-cell GLP-1 secretion

    Row 72. L cells sit in the distal small intestine and respond to
    nutrients in the lumen, so luminal_glucose is the driver rather than
    blood glucose - GLP-1 rises before the glucose it is anticipating.

    Saturating in luminal load. Steady state with the 2 min half-life in
    HormoneDegradation: ~5 pmol/L fasting, ~40-50 postprandial.

    Timescale: Minutes
    """

    inputs = {
        'luminal_glucose': ('small_intestine', 'luminal_glucose')
    }
    outputs = {
        'blood_glp1': ('blood', 'glp1')
    }

    parameters = {
        'basal_secretion': {
            'default': 1.7, 'unit': 'pmol/L/min', 'range': (0.5, 5.0),
            'description': 'Fasting L-cell secretion'
        },
        'nutrient_coupling': {
            'default': 21.0, 'unit': 'pmol/L/min', 'range': (5.0, 40.0),
            'description': 'Row 72. Maximum nutrient-driven secretion above basal'
        },
        'luminal_k': {
            'default': 15.0, 'unit': 'g', 'range': (5.0, 50.0),
            'description': 'Row 72. Half-maximal luminal glucose load'
        }
    }

    def __init__(self, basal_secretion=1.7, nutrient_coupling=21.0, luminal_k=15.0):
        super().__init__("glp1_secretion", TimeScale.MINUTES)
        self.basal_secretion = basal_secretion
        self.nutrient_coupling = nutrient_coupling
        self.luminal_k = luminal_k

    def step(self, state, dt):
        lumen = state.get_signal('small_intestine', 'luminal_glucose')
        if lumen is None or lumen < 0:
            lumen = 0.0

        secretion_rate = self.basal_secretion + \
                         self.nutrient_coupling * lumen / (self.luminal_k + lumen)

        state.update_signal('blood', 'glp1', secretion_rate * (dt/60.0))


class HormoneDegradation(ProcessModel):
    """
    First-order plasma clearance for all secreted hormones.
    Updated: adds osteocalcin and FGF23 clearance.
    """
 
    inputs = {
        'blood_insulin':           ('blood', 'insulin'),
        'blood_glucagon':          ('blood', 'glucagon'),
        'blood_erythropoietin':    ('blood', 'erythropoietin'),
        'blood_calcitonin':        ('blood', 'calcitonin'),
        'blood_parathyroid':       ('blood', 'parathyroid_hormone'),
        'blood_osteocalcin':       ('blood', 'osteocalcin'),
        'blood_fgf23':             ('blood', 'fgf23'),
        'blood_somatostatin':      ('blood', 'somatostatin'),
        'blood_amylin':            ('blood', 'amylin'),
        'blood_glp1':              ('blood', 'glp1'),
        'blood_adiponectin':       ('blood', 'adiponectin'),
        'blood_resistin':          ('blood', 'resistin'),
        'blood_cortisol':          ('blood', 'cortisol'),
        'blood_epinephrine':       ('blood', 'epinephrine'),
        'blood_norepinephrine':    ('blood', 'norepinephrine'),
        'blood_dopamine':          ('blood', 'dopamine'),
        'blood_t4':                ('blood', 't4'),
        'blood_t3':                ('blood', 't3'),
        'blood_igf1':              ('blood', 'igf1'),
        'blood_growth_hormone':    ('blood', 'growth_hormone'),
        'blood_c_peptide':         ('blood', 'c_peptide'),
        'blood_dhea':              ('blood', 'dhea'),
    }
    outputs = {
        'blood_insulin':           ('blood', 'insulin'),
        'blood_glucagon':          ('blood', 'glucagon'),
        'blood_erythropoietin':    ('blood', 'erythropoietin'),
        'blood_calcitonin':        ('blood', 'calcitonin'),
        'blood_parathyroid':       ('blood', 'parathyroid_hormone'),
        'blood_osteocalcin':       ('blood', 'osteocalcin'),
        'blood_fgf23':             ('blood', 'fgf23'),
        'blood_somatostatin':      ('blood', 'somatostatin'),
        'blood_amylin':            ('blood', 'amylin'),
        'blood_glp1':              ('blood', 'glp1'),
        'blood_adiponectin':       ('blood', 'adiponectin'),
        'blood_resistin':          ('blood', 'resistin'),
        'blood_cortisol':          ('blood', 'cortisol'),
        'blood_epinephrine':       ('blood', 'epinephrine'),
        'blood_norepinephrine':    ('blood', 'norepinephrine'),
        'blood_dopamine':          ('blood', 'dopamine'),
        'blood_t4':                ('blood', 't4'),
        'blood_t3':                ('blood', 't3'),
        'blood_igf1':              ('blood', 'igf1'),
        'blood_growth_hormone':    ('blood', 'growth_hormone'),
        'blood_c_peptide':         ('blood', 'c_peptide'),
        'blood_dhea':              ('blood', 'dhea'),
    }
 
    parameters = {
        'insulin_half_life':            {'default': 5.0,   'unit': 'minutes'},
        'glucagon_half_life':           {'default': 6.0,   'unit': 'minutes'},
        'erythropoietin_half_life':     {'default': 300.0, 'unit': 'minutes'},
        'calcitonin_half_life':         {'default': 10.0,  'unit': 'minutes'},
        'parathyroid_hormone_half_life':{'default': 4.0,   'unit': 'minutes'},
        'osteocalcin_half_life':        {'default': 300.0, 'unit': 'minutes'},  # ~5 hr
        'fgf23_half_life':              {'default': 60.0,  'unit': 'minutes'},  # ~1 hr
        'somatostatin_half_life':       {'default': 2.0,   'unit': 'minutes'},
        'amylin_half_life':             {'default': 13.0,  'unit': 'minutes'},
        'glp1_half_life':               {'default': 2.0,   'unit': 'minutes'},  # DPP-4, very short
        'adiponectin_half_life':        {'default': 150.0, 'unit': 'minutes'},  # ~2.5 hr
        'resistin_half_life':           {'default': 60.0,  'unit': 'minutes'},
        'cortisol_half_life':           {'default': 80.0,  'unit': 'minutes'},
        'epinephrine_half_life':        {'default': 2.0,   'unit': 'minutes'},
        'norepinephrine_half_life':     {'default': 2.0,   'unit': 'minutes'},
        'dopamine_half_life':           {'default': 2.0,   'unit': 'minutes'},
        't4_half_life':                 {'default': 10080.0, 'unit': 'minutes'},  # ~7 days
        't3_half_life':                 {'default': 1440.0,  'unit': 'minutes'},  # ~1 day
        'igf1_half_life':               {'default': 720.0,   'unit': 'minutes'},  # ~12 h, IGFBP-3 bound
        'growth_hormone_half_life':     {'default': 20.0,    'unit': 'minutes'},
        'c_peptide_half_life':          {'default': 30.0,    'unit': 'minutes'},
        'dhea_half_life':               {'default': 30.0,    'unit': 'minutes'},
    }
 
    def __init__(self, insulin_half_life=5.0, glucagon_half_life=6.0,
                 erythropoietin_half_life=300.0, calcitonin_half_life=10.0,
                 parathyroid_hormone_half_life=4.0, osteocalcin_half_life=300.0,
                 fgf23_half_life=60.0,  somatostatin_half_life=2.0,
                 amylin_half_life=13.0, glp1_half_life=2.0,
                 adiponectin_half_life=150.0, resistin_half_life=60.0,
                 cortisol_half_life=80.0, epinephrine_half_life=2.0,
                 norepinephrine_half_life=2.0, dopamine_half_life=2.0,
                 t4_half_life=10080.0, t3_half_life=1440.0,
                 igf1_half_life=720.0, growth_hormone_half_life=20.0,
                 c_peptide_half_life=30.0, dhea_half_life=30.0):
        super().__init__("hormone_degradation", TimeScale.MINUTES)
        self.insulin_half_life             = insulin_half_life
        self.somatostatin_half_life        = somatostatin_half_life
        self.amylin_half_life              = amylin_half_life
        self.glucagon_half_life            = glucagon_half_life
        self.erythropoietin_half_life      = erythropoietin_half_life
        self.calcitonin_half_life          = calcitonin_half_life
        self.parathyroid_hormone_half_life = parathyroid_hormone_half_life
        self.osteocalcin_half_life         = osteocalcin_half_life
        self.fgf23_half_life               = fgf23_half_life
        self.glp1_half_life                = glp1_half_life
        self.adiponectin_half_life         = adiponectin_half_life
        self.resistin_half_life            = resistin_half_life
        self.cortisol_half_life            = cortisol_half_life
        self.epinephrine_half_life         = epinephrine_half_life
        self.norepinephrine_half_life      = norepinephrine_half_life
        self.dopamine_half_life            = dopamine_half_life
        self.t4_half_life                  = t4_half_life
        self.t3_half_life                  = t3_half_life
        self.igf1_half_life                = igf1_half_life
        self.growth_hormone_half_life      = growth_hormone_half_life
        self.c_peptide_half_life           = c_peptide_half_life
        self.dhea_half_life                = dhea_half_life
 
    def _decay(self, value, half_life_min, dt_sec):
        """First-order decay: fraction removed this step."""
        return value * (1.0 - np.exp(-np.log(2) * dt_sec / (half_life_min * 60.0)))
 
    def step(self, state, dt):
        state.update_signal('blood', 'insulin',
            -self._decay(state.get_signal('blood', 'insulin'),           self.insulin_half_life, dt))
        state.update_signal('blood', 'glucagon',
            -self._decay(state.get_signal('blood', 'glucagon'),          self.glucagon_half_life, dt))
        state.update_signal('blood', 'erythropoietin',
            -self._decay(state.get_signal('blood', 'erythropoietin'),    self.erythropoietin_half_life, dt))
        state.update_signal('blood', 'calcitonin',
            -self._decay(state.get_signal('blood', 'calcitonin'),        self.calcitonin_half_life, dt))
        state.update_signal('blood', 'parathyroid_hormone',
            -self._decay(state.get_signal('blood', 'parathyroid_hormone'),self.parathyroid_hormone_half_life, dt))
        state.update_signal('blood', 'osteocalcin',
            -self._decay(state.get_signal('blood', 'osteocalcin'),       self.osteocalcin_half_life, dt))
        state.update_signal('blood', 'fgf23',
            -self._decay(state.get_signal('blood', 'fgf23'),             self.fgf23_half_life, dt))
        state.update_signal('blood', 'somatostatin',
            -self._decay(state.get_signal('blood', 'somatostatin'),      self.somatostatin_half_life, dt))
        state.update_signal('blood', 'amylin',
            -self._decay(state.get_signal('blood', 'amylin'),            self.amylin_half_life, dt))
        state.update_signal('blood', 'glp1',
            -self._decay(state.get_signal('blood', 'glp1'),              self.glp1_half_life, dt))
        state.update_signal('blood', 'adiponectin',
            -self._decay(state.get_signal('blood', 'adiponectin'),       self.adiponectin_half_life, dt))
        state.update_signal('blood', 'resistin',
            -self._decay(state.get_signal('blood', 'resistin'),          self.resistin_half_life, dt))
        state.update_signal('blood', 'cortisol',
            -self._decay(state.get_signal('blood', 'cortisol'),          self.cortisol_half_life, dt))
        state.update_signal('blood', 'epinephrine',
            -self._decay(state.get_signal('blood', 'epinephrine'),       self.epinephrine_half_life, dt))
        state.update_signal('blood', 'norepinephrine',
            -self._decay(state.get_signal('blood', 'norepinephrine'),    self.norepinephrine_half_life, dt))
        state.update_signal('blood', 'dopamine',
            -self._decay(state.get_signal('blood', 'dopamine'),          self.dopamine_half_life, dt))
        state.update_signal('blood', 't4',
            -self._decay(state.get_signal('blood', 't4'),                self.t4_half_life, dt))
        state.update_signal('blood', 't3',
            -self._decay(state.get_signal('blood', 't3'),                self.t3_half_life, dt))
        state.update_signal('blood', 'igf1',
            -self._decay(state.get_signal('blood', 'igf1'),              self.igf1_half_life, dt))
        state.update_signal('blood', 'growth_hormone',
            -self._decay(state.get_signal('blood', 'growth_hormone'),    self.growth_hormone_half_life, dt))
        state.update_signal('blood', 'c_peptide',
            -self._decay(state.get_signal('blood', 'c_peptide'),         self.c_peptide_half_life, dt))
        state.update_signal('blood', 'dhea',
            -self._decay(state.get_signal('blood', 'dhea'),              self.dhea_half_life, dt))
 

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
    Self-referential: production rate is suppressed by current EPO level
    (negative feedback term) — this process's own output feeds back into
    its own rate.
    """

    parameters = {
        'basal_epo': {'default': 10.0, 'unit': 'mU/mL', 'range': (5.0, 20.0),
                       'description': 'Baseline EPO in normoxia'},
        'max_epo': {'default': 200.0, 'unit': 'mU/mL', 'range': (100.0, 1000.0),
                     'description': 'Maximum EPO in severe hypoxia'}
    }

    def __init__(self, basal_epo=10.0, max_epo=200.0):
        super().__init__("epo_production", TimeScale.HOURS)
        self.basal_epo = basal_epo
        self.max_epo = max_epo

    def rate(self, current_epo, tissue_O2, epo_capacity, kidney_mass):
        target_O2 = 97.0
        hypoxia_factor = max(0.0, (target_O2 - tissue_O2) / target_O2)
        hill_n, K_half = 2.5, 0.15

        epo_production_rate = self.basal_epo + \
            (self.max_epo - self.basal_epo) * \
            (hypoxia_factor ** hill_n) / (K_half ** hill_n + hypoxia_factor ** hill_n)
        epo_production_rate *= epo_capacity * kidney_mass

        if current_epo > self.basal_epo * 2:
            epo_production_rate *= (self.basal_epo * 2 / current_epo)

        return epo_production_rate

    def step(self, state, dt):
        tissue_O2 = state.get_organism_state('tissue_oxygen_saturation', 95.0)
        epo_capacity = state.get_signal('kidney', 'epo_production_capacity') or 1.0
        kidney_mass = state.get_signal('kidney', 'functional_mass') or 1.0

        E0 = state.get_signal('blood', 'erythropoietin')
        if E0 is None:
            E0 = self.basal_epo

        dt_hr = dt / 3600.0

        def dEdt(t, y):
            return [self.rate(max(y[0], 0.0), tissue_O2, epo_capacity, kidney_mass)]

        sol = solve_ivp(dEdt, (0.0, dt_hr), [E0], method='RK45', rtol=1e-6, atol=1e-9)
        epo_amount = (sol.y[0, -1] - E0) if sol.success else self.rate(E0, tissue_O2, epo_capacity, kidney_mass) * dt_hr

        state.update_signal('blood', 'erythropoietin', epo_amount)


class ErythropoiesisStimulation(ProcessModel):
    """
    Bone marrow RBC production stimulated by EPO
    Self-referential: production rate is suppressed by current RBC count
    (rbc_inhibition term) — same pattern as EPO above.
    Note: atol is scaled to ~1e6-magnitude RBC counts, not the 1e-9 used
    for hormone-scale variables elsewhere — an absolute tolerance that
    tight would be meaningless (and slow) at this magnitude.
    """

    def __init__(self, basal_production=2.5e11):
        super().__init__("erythropoiesis", TimeScale.DAYS)
        self.basal_production = float(basal_production)

    def rate(self, current_rbc, epo, marrow_capacity):
        target_rbc = 5.5e6
        rbc_inhibition = target_rbc / current_rbc if current_rbc > target_rbc else 1.0

        K_epo, hill_n = 20.0, 2.0
        epo_factor = 1.0 + 3.0 * (epo ** hill_n) / (K_epo ** hill_n + epo ** hill_n)

        return self.basal_production * epo_factor * marrow_capacity * rbc_inhibition

    def step(self, state, dt):
        epo = state.get_signal('blood', 'erythropoietin') or 10.0
        marrow_capacity = state.get_signal('bone_marrow', 'erythropoiesis_capacity') or 1.0

        R0 = state.get_signal('blood', 'rbc_count')
        if R0 is None:
            R0 = 5.0e6

        dt_day = dt / 86400.0

        def dRdt(t, y):
            return [self.rate(max(y[0], 1.0), epo, marrow_capacity)]

        sol = solve_ivp(dRdt, (0.0, dt_day), [R0], method='RK45', rtol=1e-6, atol=1e3)
        new_rbcs = (sol.y[0, -1] - R0) if sol.success else self.rate(R0, epo, marrow_capacity) * dt_day

        blood_volume_liters = 5.0
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
    Self-referential: consumption rate depends on tissue glucose, which
    this process is consuming.
    """

    parameters = {
        'consumption_rate': {'default': 0.05, 'unit': 'mg/dL/min',
                              'description': 'Basal glucose consumption rate'}
    }

    def __init__(self, target_entity='muscle_tissue', consumption_rate=0.05):
        super().__init__(f"glucose_consumption_{target_entity}", TimeScale.MINUTES)
        self.target_entity = target_entity
        self.consumption_rate = consumption_rate
        self.inputs = {'tissue_glucose': (target_entity, 'glucose')}
        self.outputs = {'tissue_glucose': (target_entity, 'glucose')}

    def rate(self, tissue_glucose):
        return self.consumption_rate * (tissue_glucose / 90.0)

    def step(self, state, dt):
        T0 = state.get_signal(self.target_entity, 'glucose')
        dt_min = dt / 60.0
        if dt_min <= 0 or T0 <= 0:
            return

        def dTdt(t, y):
            return [-self.rate(max(y[0], 0.0))]

        sol = solve_ivp(dTdt, (0.0, dt_min), [T0], method='RK45', rtol=1e-6, atol=1e-9)
        amount = (T0 - sol.y[0, -1]) if sol.success else self.rate(T0) * dt_min
        amount = max(0.0, min(amount, T0))

        state.update_signal(self.target_entity, 'glucose', -amount)

class AdiponectinSecretion(ProcessModel):
    """
    White adipocyte adiponectin secretion

    Row 79 (base only - the table gives no modulator rows). Constant
    secretion scaled by adipose mass. Adiponectin is unusual among
    adipokines in falling as adiposity rises, so adipose_mass_factor
    is INVERSELY applied here and directly in ResistinSecretion.

    Long plasma half-life (~2.5 h, in HormoneDegradation), so it acts as
    a slow tonic setter of insulin sensitivity rather than a fast signal.
    Steady state ~10 µg/mL at adipose_mass_factor = 1.0. Normal range
    5-30 µg/mL.

    Timescale: Minutes
    """

    inputs = {}
    outputs = {
        'blood_adiponectin': ('blood', 'adiponectin')
    }

    parameters = {
        'basal_secretion': {
            'default': 0.0462, 'unit': 'µg/mL/min', 'range': (0.01, 0.2),
            'description': 'Row 79. Secretion rate; with the 150 min half-life this gives ~10 µg/mL steady state'
        },
        'adipose_mass_factor': {
            'default': 1.0, 'unit': 'dimensionless', 'range': (0.3, 3.0),
            'description': 'Relative adipose mass. Adiponectin falls with adiposity, so secretion is divided by this - raise it to model obesity'
        }
    }

    def __init__(self, basal_secretion=0.0462, adipose_mass_factor=1.0):
        super().__init__("adiponectin_secretion", TimeScale.MINUTES)
        self.basal_secretion = basal_secretion
        self.adipose_mass_factor = adipose_mass_factor

    def step(self, state, dt):
        rate = self.basal_secretion / max(self.adipose_mass_factor, 1e-6)
        state.update_signal('blood', 'adiponectin', rate * (dt / 60.0))


class ResistinSecretion(ProcessModel):
    """
    White adipocyte resistin secretion

    Row 81 (base only). Constant secretion scaled directly by adipose
    mass - resistin rises with adiposity, opposite to adiponectin.

    Steady state ~12 ng/mL at adipose_mass_factor = 1.0, with the 60 min
    half-life in HormoneDegradation. Normal range 7-22 ng/mL.

    Timescale: Minutes
    """

    inputs = {}
    outputs = {
        'blood_resistin': ('blood', 'resistin')
    }

    parameters = {
        'basal_secretion': {
            'default': 0.1386, 'unit': 'ng/mL/min', 'range': (0.03, 0.5),
            'description': 'Row 81. Secretion rate; with the 60 min half-life this gives ~12 ng/mL steady state'
        },
        'adipose_mass_factor': {
            'default': 1.0, 'unit': 'dimensionless', 'range': (0.3, 3.0),
            'description': 'Relative adipose mass. Resistin rises with adiposity - raise it to model obesity'
        }
    }

    def __init__(self, basal_secretion=0.1386, adipose_mass_factor=1.0):
        super().__init__("resistin_secretion", TimeScale.MINUTES)
        self.basal_secretion = basal_secretion
        self.adipose_mass_factor = adipose_mass_factor

    def step(self, state, dt):
        rate = self.basal_secretion * self.adipose_mass_factor
        state.update_signal('blood', 'resistin', rate * (dt / 60.0))


class CortisolSecretion(ProcessModel):
    """
    Adrenal zona fasciculata cortisol secretion

    Row 31 (base only - the table gives no modulator rows for cortisol).

    The physiological driver is the HPA axis (CRH -> ACTH -> cortisol), which
    is outside this table. The table instead specifies a circadian target:
    [5, 25] ug/dL in the morning, [0, 10] in the afternoon. So this is a
    circadian oscillator on time of day rather than an ACTH-driven model -
    replace the drive term if the HPA axis is ever added.

    Peak secretion ~08:00, trough ~20:00. With the 80 min half-life in
    HormoneDegradation this settles to roughly 18 ug/dL AM and 6 ug/dL PM,
    lagging the drive by about an hour as real cortisol does.

    Timescale: Minutes
    Location: Adrenal cortex, zona fasciculata (UBERON:0002054)
    """

    inputs = {
        'cortical_capacity': ('adrenal_gland', 'cortical_capacity')
    }
    outputs = {
        'blood_cortisol': ('blood', 'cortisol')
    }

    parameters = {
        'mean_secretion': {
            'default': 0.104, 'unit': 'ug/dL/min', 'range': (0.02, 0.4),
            'description': 'Row 31. Mean 24 h secretion rate; with the 80 min half-life this centres cortisol near 12 ug/dL'
        },
        'circadian_amplitude': {
            'default': 0.5, 'unit': 'fraction', 'range': (0.0, 0.9),
            'description': 'Row 31. Fractional swing about the mean; 0.5 gives roughly a 3:1 AM:PM ratio'
        },
        'peak_hour': {
            'default': 8.0, 'unit': 'hour of day', 'range': (0.0, 24.0),
            'description': 'Row 31. Clock hour of peak secretion drive'
        },
        'stress_factor': {
            'default': 1.0, 'unit': 'dimensionless', 'range': (1.0, 6.0),
            'description': 'Multiplier standing in for acute HPA activation; drive this from a perturbation to model stress'
        }
    }

    def __init__(self, mean_secretion=0.104, circadian_amplitude=0.5,
                 peak_hour=8.0, stress_factor=1.0):
        super().__init__("cortisol_secretion", TimeScale.MINUTES)
        self.mean_secretion = mean_secretion
        self.circadian_amplitude = circadian_amplitude
        self.peak_hour = peak_hour
        self.stress_factor = stress_factor

    def step(self, state, dt):
        hour_of_day = (state.time % 86400.0) / 3600.0
        phase = 2.0 * np.pi * (hour_of_day - self.peak_hour) / 24.0
        drive = 1.0 + self.circadian_amplitude * np.cos(phase)

        capacity = state.get_signal('adrenal_gland', 'cortical_capacity')
        if capacity is None:
            capacity = 1.0

        rate = self.mean_secretion * drive * capacity * self.stress_factor
        state.update_signal('blood', 'cortisol', max(0.0, rate) * (dt / 60.0))


class CatecholamineSecretion(ProcessModel):
    """
    Adrenal medulla chromaffin cell catecholamine secretion

    Rows 62 (epinephrine), 63 (norepinephrine), 64 (dopamine). All three
    are base-only rows in this table - no modulator rows are given.

    Chromaffin cells co-secrete all three from the same granules, so this is
    one process with three outputs rather than three classes, following the
    same reasoning as amylin riding with insulin.

    The table gives no driver, but organism.exercise_intensity already exists
    and the moderate_aerobic perturbation sets it, so sympathoadrenal drive is
    modelled as basal + exercise. Resting steady states are ~40 pg/mL
    epinephrine, ~250 norepinephrine, ~40 dopamine; at intensity 0.6 these
    rise roughly 10x, 6x and 3x respectively.

    NOTE: these hormones are currently WRITE-ONLY - nothing in the table reads
    them. Epinephrine is the correct driver for muscle glycogenolysis (row 60
    attributes that to glucagon, which skeletal muscle has no receptor for),
    so this is the signal to wire into GlycogenBreakdown when that is settled.

    Timescale: Minutes
    Location: Adrenal medulla, chromaffin cells
    """

    inputs = {
        'medullary_capacity': ('adrenal_gland', 'medullary_capacity'),
        'exercise_intensity': ('organism', 'exercise_intensity')
    }
    outputs = {
        'blood_epinephrine': ('blood', 'epinephrine'),
        'blood_norepinephrine': ('blood', 'norepinephrine'),
        'blood_dopamine': ('blood', 'dopamine')
    }

    parameters = {
        'basal_epinephrine': {
            'default': 13.9, 'unit': 'pg/mL/min', 'range': (3.0, 40.0),
            'description': 'Row 62. Resting secretion; with the 2 min half-life gives ~40 pg/mL'
        },
        'basal_norepinephrine': {
            'default': 86.6, 'unit': 'pg/mL/min', 'range': (20.0, 250.0),
            'description': 'Row 63. Resting secretion; gives ~250 pg/mL'
        },
        'basal_dopamine': {
            'default': 13.9, 'unit': 'pg/mL/min', 'range': (3.0, 40.0),
            'description': 'Row 64. Resting secretion; gives ~40 pg/mL'
        },
        'epinephrine_exercise_gain': {
            'default': 15.0, 'unit': 'dimensionless', 'range': (0.0, 40.0),
            'description': 'Fold increase per unit exercise_intensity. Epinephrine is the most exercise-responsive of the three'
        },
        'norepinephrine_exercise_gain': {
            'default': 8.0, 'unit': 'dimensionless', 'range': (0.0, 25.0),
            'description': 'Fold increase per unit exercise_intensity'
        },
        'dopamine_exercise_gain': {
            'default': 3.0, 'unit': 'dimensionless', 'range': (0.0, 10.0),
            'description': 'Fold increase per unit exercise_intensity; dopamine is the least responsive'
        }
    }

    def __init__(self, basal_epinephrine=13.9, basal_norepinephrine=86.6,
                 basal_dopamine=13.9, epinephrine_exercise_gain=15.0,
                 norepinephrine_exercise_gain=8.0, dopamine_exercise_gain=3.0):
        super().__init__("catecholamine_secretion", TimeScale.MINUTES)
        self.basal_epinephrine = basal_epinephrine
        self.basal_norepinephrine = basal_norepinephrine
        self.basal_dopamine = basal_dopamine
        self.epinephrine_exercise_gain = epinephrine_exercise_gain
        self.norepinephrine_exercise_gain = norepinephrine_exercise_gain
        self.dopamine_exercise_gain = dopamine_exercise_gain

    def step(self, state, dt):
        intensity = state.get_organism_state('exercise_intensity', 0.0)
        try:
            intensity = float(intensity)
        except (TypeError, ValueError):
            intensity = 0.0
        intensity = max(0.0, intensity)

        capacity = state.get_signal('adrenal_gland', 'medullary_capacity')
        if capacity is None:
            capacity = 1.0

        dt_min = dt / 60.0

        # row 62
        state.update_signal('blood', 'epinephrine',
            self.basal_epinephrine * (1.0 + self.epinephrine_exercise_gain * intensity)
            * capacity * dt_min)
        # row 63
        state.update_signal('blood', 'norepinephrine',
            self.basal_norepinephrine * (1.0 + self.norepinephrine_exercise_gain * intensity)
            * capacity * dt_min)
        # row 64
        state.update_signal('blood', 'dopamine',
            self.basal_dopamine * (1.0 + self.dopamine_exercise_gain * intensity)
            * capacity * dt_min)


class ThyroglobulinStorage(ProcessModel):
    """
    Thyroid follicular colloid thyroglobulin store

    Row 71. Thyroglobulin is the iodinated protein scaffold held in the
    follicular colloid; T4 is cleaved from it on demand, so it is a STORE
    rather than a rate. Modelled in relative units (1.0 = a fully stocked
    follicle) because the table gives no mass target.

    Synthesised toward capacity by follicular cells, depleted by T4Synthesis.
    A healthy thyroid holds weeks of hormone, so the refill constant is slow
    and the store only visibly falls under sustained high T4 output.

    Timescale: Hours
    Location: Thyroid follicular lumen, colloid
    """

    inputs = {
        'thyroglobulin': ('thyroid', 'thyroglobulin')
    }
    outputs = {
        'thyroglobulin': ('thyroid', 'thyroglobulin')
    }

    parameters = {
        'capacity': {
            'default': 1.0, 'unit': 'relative', 'range': (0.2, 1.5),
            'description': 'Row 71. Target colloid store; 1.0 = fully stocked follicle'
        },
        'refill_time_constant_hr': {
            'default': 72.0, 'unit': 'hours', 'range': (12.0, 336.0),
            'description': 'Row 71. Time constant for restocking toward capacity (~3 days)'
        }
    }

    def __init__(self, capacity=1.0, refill_time_constant_hr=72.0):
        super().__init__("thyroglobulin_storage", TimeScale.HOURS)
        self.capacity = capacity
        self.refill_time_constant_hr = refill_time_constant_hr

    def step(self, state, dt):
        tg = state.get_signal('thyroid', 'thyroglobulin')
        if tg is None:
            return
        dt_hr = dt / 3600.0
        d_tg = (self.capacity - tg) / self.refill_time_constant_hr
        state.update_signal('thyroid', 'thyroglobulin', d_tg * dt_hr)


class T4Synthesis(ProcessModel):
    """
    Thyroid follicular cell T4 (thyroxine) secretion

    Row 65. Follicular cells proteolyse colloid thyroglobulin and release T4
    into blood. Output scales with the colloid store, so a depleted follicle
    secretes less.

    The physiological driver is TSH from the anterior pituitary, which is not
    in this table (no TSH row exists), so secretion is basal rather than
    feedback-regulated - same situation as cortisol without the HPA axis.
    Replace the drive term if a TSH process is ever added.

    T4 half-life is ~7 days, handled in HormoneDegradation. Steady state
    ~8 µg/dL total T4; normal range 5-12.

    Timescale: Hours
    Location: Thyroid epithelial cell (follicular cell)
    """

    inputs = {
        'thyroglobulin': ('thyroid', 'thyroglobulin')
    }
    outputs = {
        'blood_t4': ('blood', 't4'),
        'thyroglobulin': ('thyroid', 'thyroglobulin')
    }

    parameters = {
        'basal_secretion': {
            'default': 0.033, 'unit': 'µg/dL/hr', 'range': (0.005, 0.15),
            'description': 'Row 65. T4 release rate at a full colloid store; with the 7 day half-life this gives ~8 µg/dL'
        },
        'thyroglobulin_cost': {
            'default': 0.004, 'unit': 'relative store per µg/dL T4', 'range': (0.0, 0.05),
            'description': 'Row 65. Colloid drawn down per unit T4 released; set 0.0 to make the store purely decorative'
        }
    }

    def __init__(self, basal_secretion=0.033, thyroglobulin_cost=0.004):
        super().__init__("t4_synthesis", TimeScale.HOURS)
        self.basal_secretion = basal_secretion
        self.thyroglobulin_cost = thyroglobulin_cost

    def step(self, state, dt):
        tg = state.get_signal('thyroid', 'thyroglobulin')
        if tg is None:
            tg = 1.0
        dt_hr = dt / 3600.0

        released = self.basal_secretion * max(0.0, tg) * dt_hr

        state.update_signal('blood', 't4', released)
        state.update_signal('thyroid', 'thyroglobulin', -released * self.thyroglobulin_cost)


class T4Deiodination(ProcessModel):
    """
    Peripheral and thyroidal conversion of T4 to T3 by deiodinase

    Rows 67 (thyroid, D2), 68 (kidney proximal tubule, D1),
    69 (hepatocyte, D1), 70 (skeletal myocyte, D2).

    One class, four registrations - the reaction is identical at each site and
    only the capacity differs, so site is a parameter rather than a subclass.
    Roughly 80% of circulating T3 comes from this route rather than direct
    thyroid secretion, split here liver 40% / kidney 25% / thyroid 20% /
    muscle 15%.

    First-order in T4 substrate. conversion_efficiency absorbs the molar
    weight change (T4 776.9 -> T3 651.0 g/mol) as well as the µg/dL -> ng/mL
    unit change, so it is NOT a dimensionless yield - do not read it as a
    percentage.

    Timescale: Hours
    """

    parameters = {
        'conversion_efficiency': {
            'default': 1.733e-3, 'unit': 'ng/mL T3 per µg/dL T4 per hour', 'range': (0.0, 1.0e-2),
            'description': 'Rows 67-70. Site deiodinase capacity. Defaults: liver 1.733e-3, kidney 1.083e-3, thyroid 8.664e-4, muscle 6.498e-4 - together these give ~1.2 ng/mL T3 at T4 = 8 µg/dL'
        }
    }

    def __init__(self, site='liver', conversion_efficiency=1.733e-3):
        super().__init__(f"t4_deiodination_{site}", TimeScale.HOURS)
        self.site = site
        self.conversion_efficiency = conversion_efficiency

        self.inputs = {
            'blood_t4': ('blood', 't4')
        }
        self.outputs = {
            'blood_t3': ('blood', 't3')
        }

    def step(self, state, dt):
        t4 = state.get_signal('blood', 't4')
        if t4 is None or t4 <= 0:
            return
        dt_hr = dt / 3600.0
        produced = self.conversion_efficiency * t4 * dt_hr
        state.update_signal('blood', 't3', produced)


class IGF1Secretion(ProcessModel):
    """
    Hepatocyte IGF-1 secretion

    Rows 76 (base) and 78 (fed status increases IGF-1).

    Row 76 gives age-banded targets, so basal output is looked up from the
    table's own bands rather than fitted to a curve:
        18-24 [182, 780] -> 481 ng/mL
        25-39 [114, 492] -> 303
        40-54 [90, 360]  -> 225
        55+   [71, 290]  -> 180
    Reads organism.age_years (the single canonical age field).

    The physiological driver is growth hormone acting on hepatic GHR, but the
    table gives no GH->IGF-1 row, so GH is deliberately NOT an input here.
    Adding it would close a loop with row 77 (IGF-1 -> GH), which as written
    in the table is positive feedback - see the note in GrowthHormoneSecretion.

    IGF-1 circulates bound to IGFBP-3 and has a ~12 h half-life, so it is a
    slow nutritional integrator rather than a meal-to-meal signal.

    Timescale: Minutes
    Location: Liver lobule, hepatocyte (CL:0000182)
    """

    inputs = {
        'fed_status': ('organism', 'fed_status')
    }
    outputs = {
        'blood_igf1': ('blood', 'igf1')
    }

    parameters = {
        'clearance_per_min': {
            'default': 9.627e-4, 'unit': '/min', 'range': (1e-4, 5e-3),
            'description': 'Must match igf1_half_life in HormoneDegradation (ln2/720). Used to convert the age-band target into a secretion rate.'
        },
        'fed_multiplier': {
            'default': 1.15, 'unit': 'dimensionless', 'range': (1.0, 2.0),
            'description': 'Row 78. Fed-state boost to hepatic IGF-1 output'
        },
        'fasted_multiplier': {
            'default': 0.85, 'unit': 'dimensionless', 'range': (0.3, 1.0),
            'description': 'Row 78. Fasted suppression. Prolonged fasting drops IGF-1 far more than this - lower it to model starvation.'
        },
        'age_override': {
            'default': 0.0, 'unit': 'years', 'range': (0.0, 100.0),
            'description': 'If > 0, use this age instead of reading organism state. Convenience for age sweeps.'
        }
    }

    def __init__(self, clearance_per_min=9.627e-4, fed_multiplier=1.15,
                 fasted_multiplier=0.85, age_override=0.0):
        super().__init__("igf1_secretion", TimeScale.MINUTES)
        self.clearance_per_min = clearance_per_min
        self.fed_multiplier = fed_multiplier
        self.fasted_multiplier = fasted_multiplier
        self.age_override = age_override

    def target_for_age(self, age):
        """Row 76 age bands, midpoint of each published range."""
        if age < 25:
            return 481.0
        elif age < 40:
            return 303.0
        elif age < 55:
            return 225.0
        return 180.0

    def step(self, state, dt):
        if self.age_override > 0:
            age = self.age_override
        else:
            age = state.get_organism_state('age_years', 30.0)
            try:
                age = float(age)
            except (TypeError, ValueError):
                age = 30.0

        target = self.target_for_age(age)
        secretion_rate = target * self.clearance_per_min      # ng/mL/min

        # row 78
        fed_status = state.get_organism_state('fed_status', 'fasted')
        secretion_rate *= self.fed_multiplier if fed_status == 'fed' else self.fasted_multiplier

        state.update_signal('blood', 'igf1', secretion_rate * (dt / 60.0))


class GrowthHormoneSecretion(ProcessModel):
    """
    Anterior pituitary somatotroph growth hormone secretion

    Row 77: igf-1 increases X secretion growth hormone.

    WARNING - the table's sign is opposite to established endocrinology.
    IGF-1 INHIBITS GH release, both directly at the somatotroph and by
    driving hypothalamic somatostatin; that negative feedback is the core of
    the GH/IGF-1 axis. Row 77 states "increases".

    Implemented as written, with igf1_effect POSITIVE per the table. Set
    igf1_effect to a negative value (e.g. -0.6) to get the physiological
    negative feedback. Same category of table error as rows 39-44 (muscle
    gluconeogenesis) and row 60 (glucagon on muscle glycogenolysis).

    Nothing wrote blood.growth_hormone before this process existed - it sat
    at its initial value while OsteoblastBoneFormation read it via K_gh. Bone
    formation will now respond to GH for the first time.

    GH is strongly pulsatile in vivo (nocturnal bursts to 10-30 ng/mL over a
    low baseline). The table gives no pulse row, so this is mean secretion
    only - add a circadian drive like CortisolSecretion if you need pulses.

    Timescale: Minutes
    Location: Anterior pituitary (somatotroph)
    """

    inputs = {
        'igf1': ('blood', 'igf1')
    }
    outputs = {
        'blood_growth_hormone': ('blood', 'growth_hormone')
    }

    parameters = {
        'basal_secretion': {
            'default': 0.052, 'unit': 'ng/mL/min', 'range': (0.01, 0.5),
            'description': 'Mean somatotroph output; with the 20 min half-life gives ~1.5 ng/mL'
        },
        'igf1_ref': {
            'default': 300.0, 'unit': 'ng/mL', 'range': (0.0, 600.0),
            'description': 'Row 77. IGF-1 level at which the effect is neutral'
        },
        'igf1_k': {
            'default': 200.0, 'unit': 'ng/mL', 'range': (50.0, 800.0),
            'description': 'Row 77. Half-maximal IGF-1 effect away from igf1_ref'
        },
        'igf1_effect': {
            'default': 0.6, 'unit': 'dimensionless', 'range': (-1.0, 2.0),
            'description': 'Row 77. POSITIVE = table as written (IGF-1 increases GH). NEGATIVE = physiological negative feedback. Magnitude is the maximum fractional change.'
        }
    }

    def __init__(self, basal_secretion=0.052, igf1_ref=300.0,
                 igf1_k=200.0, igf1_effect=0.6):
        super().__init__("growth_hormone_secretion", TimeScale.MINUTES)
        self.basal_secretion = basal_secretion
        self.igf1_ref = igf1_ref
        self.igf1_k = igf1_k
        self.igf1_effect = igf1_effect

    def step(self, state, dt):
        secretion_rate = self.basal_secretion

        # row 77
        igf1 = state.get_signal('blood', 'igf1')
        if igf1 is not None:
            excess = max(0.0, igf1 - self.igf1_ref)
            factor = 1.0 + self.igf1_effect * excess / (self.igf1_k + excess)
            secretion_rate *= max(0.0, factor)

        state.update_signal('blood', 'growth_hormone', secretion_rate * (dt / 60.0))


class CPeptideSecretion(ProcessModel):
    """
    Beta-cell C-peptide secretion

    Rows 27 (base) and 19 (glucose increases C-peptide).

    C-peptide is cleaved from proinsulin equimolar with insulin, so it is
    co-secreted 1:1. Row 19 names GLUCOSE as the driver rather than insulin,
    so the same Hill curve as InsulinSecretion is used here (K=90, n=1.7)
    rather than reading blood insulin the way AmylinSecretion does. Driving
    it off insulin instead would be equally defensible - swap the driver if
    you prefer the co-secretion framing.

    C-peptide is not biologically active on glucose handling; its value is as
    a marker. Because it clears far more slowly than insulin (~30 min vs
    5 min) it tracks endogenous secretion without the confound of hepatic
    first-pass extraction, which is why it distinguishes endogenous from
    injected insulin.

    Steady state ~1.2 ng/mL at glucose 90. Table target [0.5, 2.0] ng/mL.

    Timescale: Minutes
    Location: Islet of Langerhans, beta cell (CL:0000169)
    """

    inputs = {
        'glucose': ('blood', 'glucose')
    }
    outputs = {
        'blood_c_peptide': ('blood', 'c_peptide')
    }

    parameters = {
        'basal_secretion': {
            'default': 0.004, 'unit': 'ng/mL/min', 'range': (0.001, 0.02),
            'description': 'Row 27. Glucose-independent release'
        },
        'max_secretion': {
            'default': 0.05, 'unit': 'ng/mL/min', 'range': (0.02, 0.2),
            'description': 'Row 19. Maximum glucose-stimulated release; with the 30 min half-life gives ~1.2 ng/mL at glucose 90'
        },
        'glucose_sensitivity': {
            'default': 1.0, 'unit': 'dimensionless', 'range': (0.3, 2.0),
            'description': 'Row 19. Mirrors InsulinSecretion.glucose_sensitivity - move both together to keep the molar ratio sane'
        }
    }

    def __init__(self, basal_secretion=0.004, max_secretion=0.05, glucose_sensitivity=1.0):
        super().__init__("c_peptide_secretion", TimeScale.MINUTES)
        self.basal_secretion = basal_secretion
        self.max_secretion = max_secretion
        self.glucose_sensitivity = glucose_sensitivity

    def step(self, state, dt):
        glucose = state.get_signal('blood', 'glucose')
        if glucose is None or glucose <= 0:
            return

        K = 90.0   # mg/dL, same half-maximal glucose as InsulinSecretion
        n = 1.7    # same Hill coefficient - equimolar co-secretion

        stimulated = (self.max_secretion - self.basal_secretion) * \
                     (glucose**n) / (K**n + glucose**n)
        secretion_rate = self.basal_secretion + stimulated * self.glucose_sensitivity

        state.update_signal('blood', 'c_peptide', secretion_rate * (dt / 60.0))


class DHEASecretion(ProcessModel):
    """
    Adrenal zona reticularis DHEA secretion

    Row 61 (base only - no modulator rows, and the table gives no target
    range for DHEA).

    Like cortisol, the physiological driver is ACTH, which is not in this
    table, so output is basal and scaled by adrenal cortical capacity.

    Unmodelled: DHEA falls steeply with age (adrenopause - roughly 80% lower
    at 70 than at 25), and it is the one adrenal steroid with a stronger age
    dependence than circadian. organism.age_years is available if you want to
    add that; the table gives no row for it so it is left out.

    Steady state ~5 ng/mL at capacity 1.0.

    Timescale: Minutes
    Location: Adrenal cortex, zona reticularis
    """

    inputs = {
        'cortical_capacity': ('adrenal_gland', 'cortical_capacity')
    }
    outputs = {
        'blood_dhea': ('blood', 'dhea')
    }

    parameters = {
        'basal_secretion': {
            'default': 0.1155, 'unit': 'ng/mL/min', 'range': (0.02, 0.5),
            'description': 'Row 61. With the 30 min half-life this gives ~5 ng/mL'
        }
    }

    def __init__(self, basal_secretion=0.1155):
        super().__init__("dhea_secretion", TimeScale.MINUTES)
        self.basal_secretion = basal_secretion

    def step(self, state, dt):
        capacity = state.get_signal('adrenal_gland', 'cortical_capacity')
        if capacity is None:
            capacity = 1.0
        rate = self.basal_secretion * max(0.0, capacity)
        state.update_signal('blood', 'dhea', rate * (dt / 60.0))


class PancreaticBicarbonateSecretion(ProcessModel):
    """
    Pancreatic ductal bicarbonate secretion into the duodenum

    Row 26: pancreatic polypeptide decreases bicarbonate secretion.

    Note on the row as filed: the effector is given as "F cell", but F cells
    secrete PP, not bicarbonate - ductal cells do. The row means PP INHIBITS
    ductal bicarbonate output, which is the standard PP action on exocrine
    pancreas, so that is what is implemented. Output goes to the small
    intestine lumen, not to blood: this is exocrine secretion for neutralising
    gastric acid, not the plasma bicarbonate buffer.

    Basal ~0.146 mEq/min (about 1.5 L/day of juice at 140 mEq/L). Luminal
    bicarbonate is consumed by neutralisation and absorption, modelled as a
    30 min first-order loss inside this process rather than in
    HormoneDegradation, which handles plasma hormones only.

    Timescale: Minutes
    Location: Pancreatic duct -> duodenum
    """

    inputs = {
        'pancreatic_polypeptide': ('blood', 'pancreatic_polypeptide'),
        'bicarbonate': ('small_intestine', 'bicarbonate')
    }
    outputs = {
        'bicarbonate': ('small_intestine', 'bicarbonate')
    }

    parameters = {
        'basal_secretion': {
            'default': 0.146, 'unit': 'mEq/min', 'range': (0.03, 0.5),
            'description': 'Row 26. Unstimulated ductal output; ~1.5 L/day of juice at 140 mEq/L'
        },
        'pp_ref': {
            'default': 80.0, 'unit': 'pg/mL', 'range': (0.0, 200.0),
            'description': 'Row 26. PP level at which inhibition = 1.0'
        },
        'pp_k': {
            'default': 300.0, 'unit': 'pg/mL', 'range': (100.0, 1000.0),
            'description': 'Row 26. Half-maximal inhibition above pp_ref. Smaller than the islet pp_k because PP acts more strongly on exocrine pancreas than on alpha or beta cells.'
        },
        'clearance_half_life_min': {
            'default': 30.0, 'unit': 'minutes', 'range': (5.0, 180.0),
            'description': 'Neutralisation and absorption of luminal bicarbonate'
        }
    }

    def __init__(self, basal_secretion=0.146, pp_ref=80.0, pp_k=300.0,
                 clearance_half_life_min=30.0):
        super().__init__("pancreatic_bicarbonate_secretion", TimeScale.MINUTES)
        self.basal_secretion = basal_secretion
        self.pp_ref = pp_ref
        self.pp_k = pp_k
        self.clearance_half_life_min = clearance_half_life_min

    def step(self, state, dt):
        rate = self.basal_secretion

        # row 26
        pp = state.get_signal('blood', 'pancreatic_polypeptide')
        if pp is not None:
            rate *= 1.0 / (1.0 + max(0.0, pp - self.pp_ref) / self.pp_k)

        state.update_signal('small_intestine', 'bicarbonate', rate * (dt / 60.0))

        current = state.get_signal('small_intestine', 'bicarbonate')
        if current is not None and current > 0:
            lost = current * (1.0 - np.exp(-np.log(2) * dt / (self.clearance_half_life_min * 60.0)))
            state.update_signal('small_intestine', 'bicarbonate', -lost)


class FedFastedTransition(ProcessModel):
    """
    Absorptive (fed) vs post-absorptive (fasted) state machine

    Rows 12 and 21.

    DEVIATION FROM THE TABLE: rows 12 and 21 both name blood glucose as the
    trigger. Implemented here on GUT CONTENTS instead, because glucose is the
    regulated variable and using it to detect the fed state is circular - the
    better glucose homeostasis works, the less glucose moves between states,
    so the trigger degrades exactly when the model is healthy. Conversely a
    diabetic sits above any fed threshold while genuinely fasting.

    The fed state physiologically means one thing: nutrients are being
    absorbed. That is gastric contents plus luminal glucose, both of which
    already exist from the gut block. No thresholds to calibrate, and it
    works whether blood glucose is 20 or 300.

    A useful side effect: gastric emptying rate now sets how long the fed
    state lasts, and GLP-1 and amylin already slow emptying - so the incretins
    extend the absorptive phase, which is real physiology that only emerges
    if the trigger is gut contents rather than glucose.

    Set trigger_on_glucose=True to follow the table literally instead.

    THIS PROCESS OWNS organism.fed_status. The fed_status effect must be
    removed from the meal perturbations, or there will be two writers and the
    perturbation will win while it is active.

    Timescale: Minutes
    """

    inputs = {
        'gastric_contents': ('stomach', 'gastric_contents'),
        'luminal_glucose': ('small_intestine', 'luminal_glucose'),
        'glucose': ('blood', 'glucose')
    }
    outputs = {
        'fed_status': ('organism', 'fed_status')
    }

    parameters = {
        'gut_threshold_g': {
            'default': 5.0, 'unit': 'g', 'range': (0.5, 15.0),
            'description': 'Rows 12/21. Total gut carbohydrate above which the organism is absorptive. Not zero because emptying is first-order and only decays asymptotically.'
        },
        'trigger_on_glucose': {
            'default': False, 'unit': 'bool',
            'description': 'If True, follow rows 12/21 literally and switch on blood glucose instead of gut contents'
        },
        'glucose_fed_threshold': {
            'default': 100.0, 'unit': 'mg/dL', 'range': (70.0, 200.0),
            'description': 'Only used when trigger_on_glucose. Rising through this switches to fed.'
        },
        'glucose_fasted_threshold': {
            'default': 80.0, 'unit': 'mg/dL', 'range': (40.0, 120.0),
            'description': 'Only used when trigger_on_glucose. Falling through this switches to fasted. Lower than the fed threshold to give hysteresis and stop chattering at the boundary.'
        }
    }

    def __init__(self, gut_threshold_g=5.0, trigger_on_glucose=False,
                 glucose_fed_threshold=100.0, glucose_fasted_threshold=80.0):
        super().__init__("fed_fasted_transition", TimeScale.MINUTES)
        self.gut_threshold_g = gut_threshold_g
        self.trigger_on_glucose = trigger_on_glucose
        self.glucose_fed_threshold = glucose_fed_threshold
        self.glucose_fasted_threshold = glucose_fasted_threshold

    def step(self, state, dt):
        current = state.get_organism_state('fed_status', 'fasted')

        if self.trigger_on_glucose:
            glucose = state.get_signal('blood', 'glucose')
            if glucose is None:
                return
            # hysteresis: only switch on the far threshold
            if current != 'fed' and glucose >= self.glucose_fed_threshold:
                state.set_organism_state('fed_status', 'fed')          # row 21
            elif current == 'fed' and glucose <= self.glucose_fasted_threshold:
                state.set_organism_state('fed_status', 'fasted')       # row 12
            return

        stomach = state.get_signal('stomach', 'gastric_contents') or 0.0
        lumen = state.get_signal('small_intestine', 'luminal_glucose') or 0.0
        gut_load = stomach + lumen

        if gut_load > self.gut_threshold_g:
            if current != 'fed':
                state.set_organism_state('fed_status', 'fed')          # row 21
        else:
            if current != 'fasted':
                state.set_organism_state('fed_status', 'fasted')       # row 12


class MuscleGluconeogenicOutput(ProcessModel):
    """
    Skeletal muscle export of gluconeogenic substrate (Cori / glucose-alanine)

    Rows 39-44.

    DEVIATION FROM THE TABLE: all six rows say "X secretion glucose" from a
    striated muscle cell. Skeletal muscle does not express glucose-6-
    phosphatase, so it physically cannot dephosphorylate G6P and release free
    glucose - that enzyme is confined to liver, kidney and intestine. What
    muscle actually exports is LACTATE (Cori cycle) and ALANINE (glucose-
    alanine cycle); the liver then makes glucose from them.

    The modulators in the table support that reading. Insulin down (row 40),
    fed down (41), fasted up (42), cortisol up (43) and glucose down (44) are
    all correct for muscle proteolysis and lactate output. Only row 39,
    glucagon up, is wrong regardless of product - myocytes have no glucagon
    receptor, the same error as row 60.

    So this outputs lactate by default and glucagon_sensitivity defaults to
    0.0. HepaticLactateUptake then converts that lactate into blood glucose,
    which delivers what rows 39-44 intend by the route the body actually uses.

    For the literal table reading set output_substrate='glucose' and
    glucagon_sensitivity=1.0; muscle will then secrete glucose straight to
    blood.

    Timescale: Minutes
    Location: Skeletal muscle tissue, striated muscle cell (CL:0000737)
    """

    parameters = {
        'basal_output': {
            'default': 0.03466, 'unit': 'mmol/L/min', 'range': (0.005, 0.15),
            'description': 'Rows 39-44. Resting substrate export; balances HepaticLactateUptake at ~1.0 mmol/L blood lactate'
        },
        'output_substrate': {
            'default': 'lactate', 'unit': 'signal name',
            'description': "'lactate' = physiological (default). 'glucose' = literal rows 39-44, muscle secretes glucose direct to blood."
        },
        'glucagon_sensitivity': {
            'default': 0.0, 'unit': 'dimensionless', 'range': (0.0, 1.0),
            'description': 'Row 39 as written. Default 0.0 because skeletal muscle expresses no glucagon receptor. Set 1.0 to follow the table.'
        },
        'insulin_ref': {
            'default': 5.0, 'unit': 'µU/mL', 'range': (0.0, 30.0),
            'description': 'Row 40. Insulin at which suppression = 1.0'
        },
        'insulin_k': {
            'default': 30.0, 'unit': 'µU/mL', 'range': (5.0, 150.0),
            'description': 'Row 40. Half-maximal suppression of proteolysis above insulin_ref'
        },
        'fed_multiplier': {
            'default': 0.85, 'unit': 'dimensionless', 'range': (0.3, 1.0),
            'description': 'Row 41'
        },
        'fasted_multiplier': {
            'default': 1.15, 'unit': 'dimensionless', 'range': (1.0, 2.5),
            'description': 'Row 42'
        },
        'cortisol_ref': {
            'default': 10.0, 'unit': 'µg/dL', 'range': (0.0, 25.0),
            'description': 'Row 43. Matches HepaticGlucoseProduction.cortisol_ref'
        },
        'cortisol_k': {
            'default': 15.0, 'unit': 'µg/dL', 'range': (5.0, 50.0),
            'description': 'Row 43'
        },
        'cortisol_amp': {
            'default': 0.8, 'unit': 'dimensionless', 'range': (0.0, 2.0),
            'description': 'Row 43. Maximum fractional boost to proteolysis'
        },
        'glucose_ref': {
            'default': 90.0, 'unit': 'mg/dL', 'range': (40.0, 150.0),
            'description': 'Row 44'
        },
        'glucose_k': {
            'default': 60.0, 'unit': 'mg/dL', 'range': (20.0, 200.0),
            'description': 'Row 44'
        },
        'glucagon_ref': {
            'default': 60.0, 'unit': 'pg/mL', 'range': (0.0, 200.0),
            'description': 'Row 39. Only used when glucagon_sensitivity > 0'
        },
        'glucagon_k': {
            'default': 60.0, 'unit': 'pg/mL', 'range': (20.0, 300.0),
            'description': 'Row 39. Only used when glucagon_sensitivity > 0'
        }
    }

    def __init__(self, basal_output=0.03466, output_substrate='lactate',
                 glucagon_sensitivity=0.0, insulin_ref=5.0, insulin_k=30.0,
                 fed_multiplier=0.85, fasted_multiplier=1.15,
                 cortisol_ref=10.0, cortisol_k=15.0, cortisol_amp=0.8,
                 glucose_ref=90.0, glucose_k=60.0,
                 glucagon_ref=60.0, glucagon_k=60.0):
        super().__init__("muscle_gluconeogenic_output", TimeScale.MINUTES)
        self.basal_output = basal_output
        self.output_substrate = output_substrate
        self.glucagon_sensitivity = glucagon_sensitivity
        self.insulin_ref = insulin_ref
        self.insulin_k = insulin_k
        self.fed_multiplier = fed_multiplier
        self.fasted_multiplier = fasted_multiplier
        self.cortisol_ref = cortisol_ref
        self.cortisol_k = cortisol_k
        self.cortisol_amp = cortisol_amp
        self.glucose_ref = glucose_ref
        self.glucose_k = glucose_k
        self.glucagon_ref = glucagon_ref
        self.glucagon_k = glucagon_k

        self.inputs = {
            'insulin': ('blood', 'insulin'),
            'glucagon': ('blood', 'glucagon'),
            'cortisol': ('blood', 'cortisol'),
            'glucose': ('blood', 'glucose'),
            'fed_status': ('organism', 'fed_status')
        }
        self.outputs = {
            'output': ('blood', self.output_substrate)
        }

    def step(self, state, dt):
        rate = self.basal_output

        # row 40
        insulin = state.get_signal('blood', 'insulin')
        if insulin is not None:
            rate *= 1.0 / (1.0 + max(0.0, insulin - self.insulin_ref) / self.insulin_k)

        # rows 41 / 42
        fed_status = state.get_organism_state('fed_status', 'fasted')
        rate *= self.fed_multiplier if fed_status == 'fed' else self.fasted_multiplier

        # row 43
        cortisol = state.get_signal('blood', 'cortisol')
        if cortisol is not None:
            excess = max(0.0, cortisol - self.cortisol_ref)
            rate *= 1.0 + self.cortisol_amp * excess / (self.cortisol_k + excess)

        # row 44
        glucose = state.get_signal('blood', 'glucose')
        if glucose is not None:
            rate *= 1.0 / (1.0 + max(0.0, glucose - self.glucose_ref) / self.glucose_k)

        # row 39 - off by default, myocytes have no glucagon receptor
        if self.glucagon_sensitivity > 0.0:
            glucagon = state.get_signal('blood', 'glucagon')
            if glucagon is not None:
                excess = max(0.0, glucagon - self.glucagon_ref)
                boost = excess / (self.glucagon_k + excess)
                rate *= 1.0 + self.glucagon_sensitivity * boost

        state.update_signal('blood', self.output_substrate, max(0.0, rate) * (dt / 60.0))


class HepaticLactateUptake(ProcessModel):
    """
    Hepatic lactate uptake and conversion to glucose (Cori cycle)

    Not a table row - infrastructure that closes the loop opened by
    MuscleGluconeogenicOutput, in the same way IntestinalGlucoseAbsorption
    connects gastric emptying to blood glucose.

    Without it, muscle lactate output has no sink and blood lactate rises
    without bound.

    Stoichiometry: 2 lactate -> 1 glucose. One mmol/L of lactate cleared from
    5 L of blood is 5 mmol lactate -> 2.5 mmol glucose -> 450 mg -> 9 mg/dL
    across 50 dL. conversion_efficiency scales that down because a good share
    of lactate is oxidised rather than reconverted.

    Timescale: Minutes
    Location: Liver lobule, hepatocyte
    """

    inputs = {
        'lactate': ('blood', 'lactate')
    }
    outputs = {
        'lactate': ('blood', 'lactate'),
        'blood_glucose': ('blood', 'glucose')
    }

    parameters = {
        'uptake_half_life_min': {
            'default': 20.0, 'unit': 'minutes', 'range': (5.0, 120.0),
            'description': 'Blood lactate clearance half-time at rest'
        },
        'mg_dL_glucose_per_mmol_L_lactate': {
            'default': 9.0, 'unit': 'mg/dL per mmol/L', 'range': (0.0, 12.0),
            'description': 'Cori stoichiometry: 2 lactate -> 1 glucose, converted for 5 L blood into 50 dL. Do not change unless blood volume changes.'
        },
        'conversion_efficiency': {
            'default': 0.6, 'unit': 'fraction', 'range': (0.0, 1.0),
            'description': 'Fraction of cleared lactate reconverted to glucose; the remainder is oxidised'
        }
    }

    def __init__(self, uptake_half_life_min=20.0,
                 mg_dL_glucose_per_mmol_L_lactate=9.0, conversion_efficiency=0.6):
        super().__init__("hepatic_lactate_uptake", TimeScale.MINUTES)
        self.uptake_half_life_min = uptake_half_life_min
        self.mg_dL_glucose_per_mmol_L_lactate = mg_dL_glucose_per_mmol_L_lactate
        self.conversion_efficiency = conversion_efficiency

    def step(self, state, dt):
        lactate = state.get_signal('blood', 'lactate')
        if lactate is None or lactate <= 0:
            return

        cleared = lactate * (1.0 - np.exp(-np.log(2) * dt / (self.uptake_half_life_min * 60.0)))
        cleared = min(cleared, lactate)

        state.update_signal('blood', 'lactate', -cleared)
        state.update_signal('blood', 'glucose',
            cleared * self.mg_dL_glucose_per_mmol_L_lactate * self.conversion_efficiency)


class CapillaryGlucoseTransport(ProcessModel):
    """
    Perfusion-limited glucose exchange between blood and a tissue compartment

    Row 4: capillary, "transport of X from Y to Z", glucose, blood -> tissue.

    At rest, glucose delivery to most tissues is FLOW-limited rather than
    diffusion-limited, so this is a two-compartment perfusion model driven by
    the blood flow already declared in anatomy.yaml's flows: block - which
    nothing read until now.

        dC_t/dt =  (Q/V_t)(C_b - C_t)
        dC_b/dt = -(Q/V_b)(C_b - C_t)

    Mass conserving: what leaves blood arrives in the tissue.

    Integration is ANALYTIC, not Euler. The gradient obeys
    dD/dt = -kD with k = Q(1/V_b + 1/V_t), so
        transferred = D0 (1 - exp(-k dt)) / (1/V_b + 1/V_t)
    With liver perfusion k is about 1.17/min, a time constant SHORTER than the
    60 s timestep - an Euler step would overshoot and ring. The closed form is
    exact for any dt.

    WHICH COMPARTMENTS: register this only for tissues that have no uptake
    process. muscle_tissue, adipose_tissue and heart are all served by
    GlucoseUptake, which already moves glucose out of blood into them;
    adding transport there would move the same glucose twice. That leaves
    liver and brain - and brain is a good fit, since its GLUT1/GLUT3 uptake
    is insulin-independent and genuinely perfusion-limited.

    Timescale: Seconds in the table; registered at minutes here because the
    engine gives SECONDS processes dt=1.0 while advancing 60 s of sim time.

    Location: Capillary (UBERON:0001982)
    """

    parameters = {
        'flow_L_per_min': {
            'default': 0.0, 'unit': 'L/min', 'range': (0.0, 5.0),
            'description': 'Perfusion. If 0.0, read from the anatomy flows: block entry named by flow_id. Set explicitly to override.'
        },
        'target_volume_L': {
            'default': 0.0, 'unit': 'L', 'range': (0.0, 10.0),
            'description': 'Tissue distribution volume. If 0.0, read from the organ volume in anatomy.'
        },
        'blood_volume_L': {
            'default': 5.0, 'unit': 'L', 'range': (3.0, 7.0),
            'description': 'Blood compartment volume'
        },
        'extraction_efficiency': {
            'default': 1.0, 'unit': 'fraction', 'range': (0.0, 1.0),
            'description': 'Fraction of the perfusion-limited flux actually realised. 1.0 = fully flow-limited; lower it to make exchange partly diffusion-limited.'
        }
    }

    def __init__(self, target_organ='liver', flow_id=None, flow_L_per_min=0.0,
                 target_volume_L=0.0, blood_volume_L=5.0, extraction_efficiency=1.0):
        super().__init__(f"capillary_glucose_transport_{target_organ}", TimeScale.MINUTES)
        self.target_organ = target_organ
        self.flow_id = flow_id or f"blood_to_{target_organ}"
        self.flow_L_per_min = flow_L_per_min
        self.target_volume_L = target_volume_L
        self.blood_volume_L = blood_volume_L
        self.extraction_efficiency = extraction_efficiency

        self.inputs = {
            'blood_glucose': ('blood', 'glucose'),
            'target_glucose': (target_organ, 'glucose')
        }
        self.outputs = {
            'blood_glucose': ('blood', 'glucose'),
            'target_glucose': (target_organ, 'glucose')
        }

    def _resolve(self, state):
        """Pull flow and volume from anatomy unless explicitly overridden."""
        Q = self.flow_L_per_min
        if Q <= 0.0:
            flow = state.flows.get(self.flow_id)
            if flow:
                Q = flow.get('rate', 0.0)

        V_t = self.target_volume_L
        if V_t <= 0.0:
            organ = state.organs.get(self.target_organ)
            if organ:
                V_t = organ.get('volume', 0.0)
        return Q, V_t

    def step(self, state, dt):
        Q, V_t = self._resolve(state)
        if Q <= 0.0 or V_t <= 0.0:
            return

        C_b = state.get_signal('blood', 'glucose')
        C_t = state.get_signal(self.target_organ, 'glucose')
        if C_b is None or C_t is None:
            return

        V_b = self.blood_volume_L
        gradient = C_b - C_t
        if gradient == 0.0:
            return

        inv_sum = 1.0 / V_b + 1.0 / V_t
        k = Q * inv_sum                      # per minute
        dt_min = dt / 60.0

        # exact solution of dD/dt = -kD, expressed as mass moved
        transferred = gradient * (1.0 - np.exp(-k * dt_min)) / inv_sum
        transferred *= self.extraction_efficiency

        state.update_signal('blood', 'glucose', -transferred / V_b)
        state.update_signal(self.target_organ, 'glucose', transferred / V_t)