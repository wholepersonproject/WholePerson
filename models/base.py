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
    
    Subclasses should declare inputs, outputs, and parameters:
    
    Example:
        inputs = {
            'glucose': ('blood', 'glucose'),
            'insulin': ('blood', 'insulin')
        }
        
        outputs = {
            'epo': ('blood', 'erythropoietin')
        }
        
        parameters = {
            'glucose_sensitivity': {
                'default': 1.0,
                'unit': 'dimensionless',
                'range': (0.3, 2.0),
                'description': 'Beta-cell responsiveness'
            }
        }
    """
    
    # Subclasses declare these
    inputs = {}
    outputs = {}
    parameters = {}  # NEW: Parameter metadata
    
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
    
    # NEW: Get parameter metadata
    @classmethod
    def get_parameter_info(cls):
        """Return parameter metadata for this process"""
        return cls.parameters
    
    def can_execute(self, state):
        """
        Check if all required inputs exist in state
        Uses smart lookup: checks entities, organs, and tissues
        
        Returns:
            (bool, list): (can_run, missing_inputs)
        """
        missing = []
        
        for input_name, location in self.inputs.items():
            if len(location) == 2:
                target_id, signal_name = location
                
                found = False
                
                if target_id in state.entities:
                    if state.has_entity_signal(target_id, signal_name):
                        found = True
                elif target_id in state.organs:
                    if state.has_organ_signal(target_id, signal_name):
                        found = True
                elif target_id in state.tissues:
                    if state.has_tissue_signal(target_id, signal_name):
                        found = True
                
                if not found:
                    missing.append(f"{target_id}.{signal_name}")
            
            elif len(location) == 3:
                if location[2] == 'organ':
                    organ_id, signal_name = location[0], location[1]
                    if not state.has_organ_signal(organ_id, signal_name):
                        missing.append(f"organ.{organ_id}.{signal_name}")
                
                elif location[2] == 'organism':
                    state_name = location[0]
                    if not state.has_organism_state(state_name):
                        missing.append(f"organism.{state_name}")
        
        return len(missing) == 0, missing
    
    def validate_outputs(self, state):
        """
        Check if output targets (entities/organs/tissues) exist
        
        Returns:
            (bool, list): (can_write, missing_targets)
        """
        missing_targets = []
        
        for output_name, location in self.outputs.items():
            if len(location) == 2:
                target_id, signal_name = location
                
                if (target_id not in state.entities and 
                    target_id not in state.organs and 
                    target_id not in state.tissues):
                    missing_targets.append(f"{target_id}")
            
            elif len(location) == 3:
                if location[2] == 'organ':
                    organ_id = location[0]
                    if organ_id not in state.organs:
                        missing_targets.append(f"organ.{organ_id}")
                
                elif location[2] == 'organism':
                    pass
        
        return len(missing_targets) == 0, missing_targets
    
    @abstractmethod
    def step(self, state, dt):
        """
        Execute one timestep of the process
        
        Args:
            state: SimulationState object
            dt: Timestep in seconds
        """
        pass