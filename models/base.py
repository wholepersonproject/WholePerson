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
    
    Subclasses should declare inputs and outputs for clarity and validation:
    
    Example:
        inputs = {
            'glucose': ('blood', 'glucose'),                              # entity signal
            'insulin': ('blood', 'insulin'),                              # entity signal
            'epo_capacity': ('kidney', 'epo_production_capacity', 'organ'), # organ signal
            'fed_status': ('fed_status', None, 'organism')                # organism state
        }
        
        outputs = {
            'epo': ('blood', 'erythropoietin'),                           # entity signal
            'kidney_epo': ('kidney', 'epo_stored', 'organ')               # organ signal
        }
    
    Input/Output format:
        - Entity signal: (entity_id, signal_name)
        - Organ signal:  (organ_id, signal_name, 'organ')
        - Organism state: (state_name, None, 'organism')
    """
    
    # Subclasses declare these (recommended for automatic validation)
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
    
    def can_execute(self, state):
        """
        Check if all required inputs exist in state
        Uses smart lookup: checks entities, organs, and tissues
        
        Returns:
            (bool, list): (can_run, missing_inputs)
        
        Examples:
            can_run, missing = model.can_execute(state)
            if not can_run:
                print(f"Cannot run: missing {missing}")
        """
        missing = []
        
        for input_name, location in self.inputs.items():
            if len(location) == 2:
                # Smart lookup: check entities, organs, and tissues
                target_id, signal_name = location
                
                # Check if target exists anywhere and has the signal
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
                # Organ signal or organism state
                if location[2] == 'organ':
                    # ('kidney', 'epo_capacity', 'organ')
                    organ_id, signal_name = location[0], location[1]
                    if not state.has_organ_signal(organ_id, signal_name):
                        missing.append(f"organ.{organ_id}.{signal_name}")
                
                elif location[2] == 'organism':
                    # ('fed_status', None, 'organism')
                    state_name = location[0]
                    if not state.has_organism_state(state_name):
                        missing.append(f"organism.{state_name}")
        
        return len(missing) == 0, missing
    
    def validate_outputs(self, state):
        """
        Check if output targets (entities/organs/tissues) exist
        Uses smart lookup: checks entities, organs, and tissues
        Note: Signals will be auto-created if the target exists
        
        Returns:
            (bool, list): (can_write, missing_targets)
        
        Examples:
            can_write, missing = model.validate_outputs(state)
            if not can_write:
                print(f"Cannot write: missing targets {missing}")
        """
        missing_targets = []
        
        for output_name, location in self.outputs.items():
            if len(location) == 2:
                # Smart lookup: check if target exists in entities, organs, or tissues
                target_id, signal_name = location
                
                if (target_id not in state.entities and 
                    target_id not in state.organs and 
                    target_id not in state.tissues):
                    missing_targets.append(f"{target_id}")
            
            elif len(location) == 3:
                if location[2] == 'organ':
                    # ('kidney', 'epo_stored', 'organ')
                    organ_id = location[0]
                    if organ_id not in state.organs:
                        missing_targets.append(f"organ.{organ_id}")
                
                elif location[2] == 'organism':
                    # Organism state always exists, nothing to validate
                    pass
        
        return len(missing_targets) == 0, missing_targets
    
    @abstractmethod
    def step(self, state, dt):
        """
        Execute one timestep of the process
        
        Args:
            state: SimulationState object
            dt: Timestep in seconds
        
        Note: 
            - Inputs are guaranteed to exist (checked by engine)
            - Output targets are guaranteed to exist (checked by engine)
            - Output signals will be auto-created if needed
        """
        pass