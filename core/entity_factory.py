import yaml
import numpy as np
from importlib import import_module

class EntityFactory:
    """Factory for building simulation entities from configuration"""
    
    def __init__(self, schema_path):
        with open(schema_path, 'r') as f:
            self.schema = yaml.safe_load(f)
    
    def initialize_simulation_state(self, state):
        """
        Initialize state from anatomy schema
        NEW: Also extracts and registers constraints from signal definitions
        """
        # Create organs first (they may be referenced by entities)
        for organ_id, config in self.schema.get('organs', {}).items():
            self._create_organ(state, organ_id, config)
        
        # Create tissues
        for tissue_id, config in self.schema.get('tissues', {}).items():
            self._create_tissue(state, tissue_id, config)
        
        # Create entities (fluids, cell populations, etc.)
        for entity_id, config in self.schema.get('entities', {}).items():
            self._create_entity(state, entity_id, config)
        
        # Create flows
        for flow_id, config in self.schema.get('flows', {}).items():
            state.add_flow(
                flow_id,
                config['from'],
                config['to'],
                config['rate'],
                config.get('type', 'transport')
            )
        
        # Set organism state
        for state_name, state_config in self.schema.get('organism', {}).items():
            if isinstance(state_config, dict):
                # New format with constraints
                state.set_organism_state(state_name, state_config.get('initial', 0.0))
                self._register_constraint(state, 'organism', state_name, state_config)
            else:
                # Old format (just a value)
                state.set_organism_state(state_name, state_config)
        
        print(f"✓ Initialized:")
        print(f"  Entities: {len(state.entities)}")
        print(f"  Organs: {len(state.organs)}")
        print(f"  Tissues: {len(state.tissues)}")
        print(f"  Flows: {len(state.flows)}")
        if hasattr(state, '_constraints'):
            print(f"  Constraints: {len(state._constraints)}")
    
    def _create_organ(self, state, organ_id, config):
        """Create an organ in state.organs (can be lumped or spatial)"""
        representation = config.get('representation', 'lumped')
        organ_system = config.get('organ_system', 'unknown')
        
        if representation == 'lumped':
            # Extract signals and constraints
            signals = {}
            for signal_name, signal_config in config.get('signals', {}).items():
                if isinstance(signal_config, dict):
                    # New format with metadata
                    signals[signal_name] = signal_config.get('initial', 0.0)
                    self._register_constraint(state, organ_id, signal_name, signal_config)
                else:
                    # Old format (just a value)
                    signals[signal_name] = signal_config
            
            state.add_organ(
                organ_id,
                organ_system,
                representation=representation,
                spatial_type=config.get('spatial_type', 'localized'),
                position=config.get('position'),
                volume=config.get('volume', 1.0),
                geometry=config.get('geometry'),
                signals=signals
            )
        
        elif representation == 'spatial':
            # Convert scalar signals to spatial fields
            shape = tuple(config['shape'])
            spatial_signals = {}
            
            for signal_name, signal_config in config.get('signals', {}).items():
                if isinstance(signal_config, dict):
                    # New format with metadata
                    value = signal_config.get('initial', 0.0)
                    self._register_constraint(state, organ_id, signal_name, signal_config)
                else:
                    # Old format
                    value = signal_config
                
                if isinstance(value, (int, float)):
                    # Convert scalar to uniform field
                    spatial_signals[signal_name] = np.full(shape, value, dtype=np.float32)
                else:
                    # Already a field
                    spatial_signals[signal_name] = value
            
            state.add_organ(
                organ_id,
                organ_system,
                representation=representation,
                spatial_type=config.get('spatial_type', 'localized'),
                position=config.get('position'),
                shape=shape,
                dx=config.get('dx', 1.0),
                volume=config.get('volume', 1.0),
                geometry=config.get('geometry'),
                signals=spatial_signals
            )
    
    def _create_tissue(self, state, tissue_id, config):
        """Create a tissue in state.tissues"""
        tissue_type = config.get('type', 'generic')
        
        # Extract signals and constraints
        signals = {}
        for signal_name, signal_config in config.get('signals', {}).items():
            if isinstance(signal_config, dict):
                signals[signal_name] = signal_config.get('initial', 0.0)
                self._register_constraint(state, tissue_id, signal_name, signal_config)
            else:
                signals[signal_name] = signal_config
        
        state.add_tissue(
            tissue_id,
            tissue_type,
            signals=signals
        )
    
    def _create_entity(self, state, entity_id, config):
        """Create an entity in state.entities (fluids, cell populations)"""
        entity_type = config['type']
        representation = config['representation']
        spatial_type = config.get('spatial_type', 'localized')
        
        if representation == 'lumped':
            # Extract signals and constraints
            signals = {}
            for signal_name, signal_config in config.get('signals', {}).items():
                if isinstance(signal_config, dict):
                    signals[signal_name] = signal_config.get('initial', 0.0)
                    self._register_constraint(state, entity_id, signal_name, signal_config)
                else:
                    signals[signal_name] = signal_config
            
            state.add_entity(
                entity_id,
                entity_type,
                representation,
                spatial_type=spatial_type,
                position=config.get('position'),
                volume=config.get('volume', 1.0),
                signals=signals
            )
        
        elif representation == 'spatial':
            # Extract signals and constraints
            signals = {}
            for signal_name, signal_config in config.get('signals', {}).items():
                if isinstance(signal_config, dict):
                    signals[signal_name] = signal_config.get('initial', 0.0)
                    self._register_constraint(state, entity_id, signal_name, signal_config)
                else:
                    signals[signal_name] = signal_config
            
            state.add_entity(
                entity_id,
                entity_type,
                representation,
                spatial_type=spatial_type,
                position=config.get('position'),
                shape=tuple(config['shape']),
                dx=config.get('dx', 1.0),
                volume=config.get('volume', 1.0),
                signals=signals
            )
        
        elif representation == 'agents':
            state.add_entity(
                entity_id,
                entity_type,
                representation,
                spatial_type=spatial_type,
                parent_entity=config.get('parent_entity'),
                count=config.get('count', 0)
            )
            
            count = config.get('count', 0)
            parent = config.get('parent_entity')
            distribution = config.get('distribution', 'random')
            agent_props = config.get('agent_properties', {})
            
            for i in range(count):
                agent_id = f"{entity_id}_{i}"
                position = self._generate_position(state, parent, distribution)
                
                state.add_agent(
                    entity_id,
                    agent_id,
                    position=position,
                    state=agent_props.copy()
                )
    
    def _generate_position(self, state, parent_entity_id, distribution):
        """Generate position for agent-based entities"""
        if not parent_entity_id:
            return None
        
        # Check if parent is an organ
        if parent_entity_id in state.organs:
            # Organs don't have spatial info yet, return None
            return None
        
        # Check if parent is an entity
        if parent_entity_id not in state.entities:
            return None
        
        parent = state.entities[parent_entity_id]
        
        if parent['representation'] == 'spatial':
            shape = parent['shape']
            if distribution == 'random':
                return tuple(np.random.randint(0, dim) for dim in shape)
        
        return None
    
    def _register_constraint(self, state, target_id, signal_name, config):
        """
        Extract and register constraint from signal config
        NEW method for constraint extraction
        
        Args:
            state: SimulationState object
            target_id: Entity/organ/tissue/organism ID
            signal_name: Signal name
            config: Signal configuration dict
        """
        # Only register if state has constraint system
        if not hasattr(state, 'add_constraint'):
            return
        
        constraint = {}
        
        if 'min' in config:
            constraint['min'] = config['min']
        if 'max' in config:
            constraint['max'] = config['max']
        if 'warn_below' in config:
            constraint['warn_below'] = config['warn_below']
        if 'warn_above' in config:
            constraint['warn_above'] = config['warn_above']
        if 'unit' in config:
            constraint['unit'] = config['unit']
        
        if constraint:
            state.add_constraint(target_id, signal_name, **constraint)


class ProcessLoader:
    """Load physiological processes from registry configuration"""
    
    def __init__(self, registry_path):
        with open(registry_path, 'r') as f:
            self.registry = yaml.safe_load(f)
    
    def load_all_processes(self, engine):
        """Load all processes from registry"""
        processes = self.registry['processes']
        
        for process_id, config in processes.items():
            # Parse module and class
            module_path, class_name = config['class'].rsplit('.', 1)
            module = import_module(module_path)
            ModelClass = getattr(module, class_name)
            
            # Get parameters
            params = config.get('parameters', {})
            
            # Instantiate
            model = ModelClass(**params)
            
            # Get dependencies
            dependencies = config.get('dependencies', [])
            
            # Register
            engine.register_model(process_id, model, dependencies)
        
        print(f"✓ Loaded {len(engine.models)} processes")
    
    def load_specific_processes(self, engine, process_ids):
        """Load only specific processes"""
        processes = self.registry['processes']
        
        for process_id in process_ids:
            if process_id not in processes:
                print(f"⚠️  Process '{process_id}' not found in registry")
                continue
            
            config = processes[process_id]
            module_path, class_name = config['class'].rsplit('.', 1)
            module = import_module(module_path)
            ModelClass = getattr(module, class_name)
            
            params = config.get('parameters', {})
            model = ModelClass(**params)
            dependencies = config.get('dependencies', [])
            
            engine.register_model(process_id, model, dependencies)
        
        print(f"✓ Loaded {len(process_ids)} specific processes")