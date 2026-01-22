from abc import ABC, abstractmethod
from enum import Enum

class TimeScale(Enum):
    """Timescale in seconds"""
    SECONDS = 1.0
    MINUTES = 60.0
    HOURS = 3600.0
    DAYS = 86400.0
    MONTHS = 2628000.0
    
class ProcessModel(ABC):
    """
    Base class for physiological processes
    
    Subclasses should declare inputs and outputs for clarity:
    
    inputs = {
        'glucose': ('blood', 'glucose'),
        'insulin': ('blood', 'insulin')
    }
    
    outputs = {
        'uptake': ('liver', 'glucose_uptake')
    }
    """
    
    # Subclasses can declare these (optional but recommended)
    inputs = {}
    outputs = {}
    
    def __init__(self, process_id, timescale):
        self.process_id = process_id
        self.timescale = timescale
    
    def get_interface(self):
        """Return process interface (inputs/outputs)"""
        return {
            'process_id': self.process_id,
            'timescale': self.timescale.name,
            'inputs': self.inputs,
            'outputs': self.outputs
        }
    
    @abstractmethod
    def step(self, state, dt):
        pass

