import numpy as np
from models.base import ProcessModel, TimeScale
class RenalGlucoseExcretion(ProcessModel):
    """
    Renal glucose reabsorption / excretion (proximal tubule)

    Mechanism: SGLT2/SGLT1 (apical) + GLUT2 (basolateral) reabsorb filtered
    glucose, capacity-limited by Transport Maximum (Tm). Below Tm, filtered
    glucose is essentially fully reabsorbed back to blood (zero excretion).
    Above Tm, the excess passes into urine one-for-one (glucosuria).

    Uses a softplus function to smooth the corner at Tm ("splay" — the
    gradual, not step-function, onset of glucosuria across the nephron
    population), rather than a Michaelis-Menten-style saturation curve.
    softplus is correct here because it is exactly zero well below Tm and
    exactly linear (slope 1) well above it — a MM curve is NOT, and leaks
    substantially even at normal glucose (see prior discussion).

    excreted(L) = S * ln(1 + exp((L - Tm) / S))
    reabsorbed(L) = L - excreted(L)

    where L = filtered load = GFR * blood_glucose, S = splay_width
    (smaller S = sharper corner, closer to a hard min(L, Tm) cutoff).

    GFR is read as a signal, not a fixed constructor parameter — it's a
    time-varying quantity driven by its own physiology (arteriolar tone,
    autoregulation, sympathetic activity, RAAS/ANP — Urinary-system table
    rows 52-76, 104), which belongs in a separate process. This model
    just consumes whatever GFR is currently in state, with a fallback if
    no GFR-producing process is registered yet.

    Timescale: Minutes
    Integration: direct rate*dt (no solve_ivp) — see prior discussion on
    why RK4 doesn't meaningfully differ from Euler here.

    Source: Whole Person Physiome Table, Urinary-system_v1.0, rows
    66/86/87 (SGLT2/SGLT1 apical uptake + GLUT2 basolateral export,
    Tm/splay mechanism).
    """

    inputs = {
        'blood_glucose': ('blood', 'glucose'),
        'gfr': ('kidney', 'gfr')
    }
    outputs = {
        'blood_glucose': ('blood', 'glucose'),
        'urinary_glucose': ('kidney', 'urinary_glucose')
    }

    parameters = {
        'tm_mg_per_min': {
            'default': 225.0,
            'unit': 'mg/min',
            'range': (150.0, 300.0),
            'description': 'Renal glucose transport maximum. Default corresponds to a ~180 mg/dL renal threshold at fallback_gfr_dl_per_min=1.25 (Tm = threshold * GFR)'
        },
        'splay_width_mg_per_min': {
            'default': 8.0,
            'unit': 'mg/min',
            'range': (2.0, 40.0),
            'description': 'Sharpness of the onset of glucosuria near Tm. Small (~5-10) = close to a hard cutoff; large (~30-40) = gradual leak starting well below Tm.'
        },
        'fallback_gfr_dl_per_min': {
            'default': 1.25,
            'unit': 'dL/min',
            'range': (0.5, 1.5),
            'description': 'Used only if no GFR-producing process has set kidney.gfr yet'
        }
    }

    def __init__(self, tm_mg_per_min=225.0, splay_width_mg_per_min=8.0, fallback_gfr_dl_per_min=1.25):
        super().__init__("renal_glucose_excretion", TimeScale.MINUTES)
        self.tm_mg_per_min = tm_mg_per_min
        self.splay_width_mg_per_min = splay_width_mg_per_min
        self.fallback_gfr_dl_per_min = fallback_gfr_dl_per_min

    def step(self, state, dt):
        blood_glucose = state.get_signal('blood', 'glucose')
        if blood_glucose is None or blood_glucose <= 0:
            return

        gfr = state.get_signal('kidney', 'gfr')
        if gfr is None or gfr <= 0:
            gfr = self.fallback_gfr_dl_per_min

        dt_min = dt / 60.0

        filtered_load = gfr * blood_glucose  # mg/min

        S = self.splay_width_mg_per_min
        z = (filtered_load - self.tm_mg_per_min) / S
        # clip to avoid overflow in exp() for very high glucose
        z = min(z, 50.0)
        excreted_mg_per_min = S * np.log1p(np.exp(z))

        excreted_mg = excreted_mg_per_min * dt_min

        blood_volume_dL = 50.0
        glucose_drop_mg_per_dL = min(excreted_mg / blood_volume_dL, blood_glucose)

        state.update_signal('blood', 'glucose', -glucose_drop_mg_per_dL)
        state.update_signal('kidney', 'urinary_glucose', excreted_mg)