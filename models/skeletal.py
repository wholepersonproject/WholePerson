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
