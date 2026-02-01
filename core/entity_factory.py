import yaml
import numpy as np
from importlib import import_module

class EntityFactory:
    def __init__(self, schema_path):
        with open(schema_path, 'r') as f:
            self.schema = yaml.safe_load(f)
    
    def initialize_simulation_state(self, state):
        """Initialize state from anatomy schema"""
        
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
        for state_name, value in self.schema.get('organism', {}).items():
            state.set_organism_state(state_name, value)
        
        print(f"✓ Initialized:")
        print(f"  Entities: {len(state.entities)}")
        print(f"  Organs: {len(state.organs)}")
        print(f"  Tissues: {len(state.tissues)}")
        print(f"  Flows: {len(state.flows)}")
    
    def _create_organ(self, state, organ_id, config):
        """Create an organ in state.organs (can be lumped or spatial)"""
        representation = config.get('representation', 'lumped')
        organ_system = config.get('organ_system', 'unknown')
        
        if representation == 'lumped':
            state.add_organ(
                organ_id,
                organ_system,  # This is organ_type parameter
                representation=representation,
                spatial_type=config.get('spatial_type', 'localized'),
                position=config.get('position'),
                volume=config.get('volume', 1.0),
                signals=config.get('signals', {})
            )
        
        elif representation == 'spatial':
            # Convert scalar signals to spatial fields
            shape = tuple(config['shape'])
            spatial_signals = {}
            for signal_name, value in config.get('signals', {}).items():
                if isinstance(value, (int, float)):
                    # Convert scalar to uniform field
                    spatial_signals[signal_name] = np.full(shape, value, dtype=np.float32)
                else:
                    # Already a field
                    spatial_signals[signal_name] = value
            
            state.add_organ(
                organ_id,
                organ_system,  # This is organ_type parameter
                representation=representation,
                spatial_type=config.get('spatial_type', 'localized'),
                position=config.get('position'),
                shape=shape,
                dx=config.get('dx', 1.0),
                volume=config.get('volume', 1.0),
                signals=spatial_signals
            )
    
    def _create_tissue(self, state, tissue_id, config):
        """Create a tissue in state.tissues"""
        tissue_type = config.get('type', 'generic')  # Default to 'generic' if not specified
        
        state.add_tissue(
            tissue_id,
            tissue_type,
            signals=config.get('signals', {})
        )
    
    def _create_entity(self, state, entity_id, config):
        """Create an entity in state.entities (fluids, cell populations)"""
        entity_type = config['type']
        representation = config['representation']
        spatial_type = config.get('spatial_type', 'localized')
        
        if representation == 'lumped':
            state.add_entity(
                entity_id,
                entity_type,
                representation,
                spatial_type=spatial_type,
                position=config.get('position'),
                volume=config.get('volume', 1.0),
                signals=config.get('signals', {})
            )
        
        elif representation == 'spatial':
            state.add_entity(
                entity_id,
                entity_type,
                representation,
                spatial_type=spatial_type,
                position=config.get('position'),
                shape=tuple(config['shape']),
                dx=config.get('dx', 1.0),
                volume=config.get('volume', 1.0),
                signals=config.get('signals', {})
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


class ProcessLoader:
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