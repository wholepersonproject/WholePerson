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
    """Base class for all physiological processes"""
    
    def __init__(self, process_id, timescale):
        self.process_id = process_id
        self.timescale = timescale
    
    @abstractmethod
    def step(self, state, dt):
        """Execute process. Read from state, compute, write to state."""
        pass

