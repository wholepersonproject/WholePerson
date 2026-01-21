import numpy as np

class SimulationState:
    '''Unified state with flexible entity representations'''
    
    def __init__(self):
        self.time = 0.0
        self.entities = {}
        self.flows = {}
        self.organism = {}
        self.organ_systems = {}  # NEW: Track system-wide states
        self.history = []
        self._engine = None  # Set by engine for perturbation access
    
    # =========================================================================
    # ENTITY MANAGEMENT
    # =========================================================================
    
    def add_entity(self, entity_id, entity_type, representation, **kwargs):
        self.entities[entity_id] = {
            'type': entity_type,
            'representation': representation,
            'spatial_type': kwargs.get('spatial_type', 'localized'),
            'position': kwargs.get('position'),
            **kwargs
        }
        
        if representation == 'lumped':
            self.entities[entity_id]['signals'] = kwargs.get('signals', {})
        
        elif representation == 'spatial':
            shape = kwargs['shape']
            self.entities[entity_id]['shape'] = shape
            self.entities[entity_id]['dx'] = kwargs.get('dx', 1.0)
            self.entities[entity_id]['signals'] = {}
            for signal_name, initial_value in kwargs.get('signals', {}).items():
                self.entities[entity_id]['signals'][signal_name] = \
                    np.full(shape, initial_value, dtype=np.float32)
        
        elif representation == 'agents':
            self.entities[entity_id]['parent_entity'] = kwargs.get('parent_entity')
            self.entities[entity_id]['count'] = kwargs.get('count', 0)
            self.entities[entity_id]['agents'] = []
    
    def add_flow(self, flow_id, from_entity, to_entity, rate, flow_type='transport'):
        self.flows[flow_id] = {
            'from': from_entity,
            'to': to_entity,
            'rate': rate,
            'type': flow_type
        }
    
    # =========================================================================
    # SIGNAL ACCESS (unified API for all representations)
    # =========================================================================
    
    def get_signal(self, entity_id, signal_name):
        entity = self.entities[entity_id]
        rep = entity['representation']
        
        if rep == 'lumped':
            return entity['signals'].get(signal_name, 0.0)
        elif rep == 'spatial':
            field = entity['signals'].get(signal_name)
            return np.mean(field) if field is not None else 0.0
        elif rep == 'agents':
            values = [agent['state'].get(signal_name, 0.0) 
                     for agent in entity['agents']]
            return np.mean(values) if values else 0.0
        return 0.0
    
    def update_signal(self, entity_id, signal_name, delta):
        entity = self.entities[entity_id]
        rep = entity['representation']
        
        if rep == 'lumped':
            current = entity['signals'].get(signal_name, 0.0)
            entity['signals'][signal_name] = current + delta
        elif rep == 'spatial':
            if signal_name in entity['signals']:
                entity['signals'][signal_name] += delta
        elif rep == 'agents':
            for agent in entity['agents']:
                current = agent['state'].get(signal_name, 0.0)
                agent['state'][signal_name] = current + delta
    
    def set_signal(self, entity_id, signal_name, value):
        entity = self.entities[entity_id]
        rep = entity['representation']
        
        if rep == 'lumped':
            entity['signals'][signal_name] = value
        elif rep == 'spatial':
            if signal_name in entity['signals']:
                entity['signals'][signal_name][:] = value
        elif rep == 'agents':
            for agent in entity['agents']:
                agent['state'][signal_name] = value
    
    # =========================================================================
    # SPATIAL-SPECIFIC
    # =========================================================================
    
    def get_signal_field(self, entity_id, signal_name):
        entity = self.entities[entity_id]
        if entity['representation'] != 'spatial':
            raise ValueError(f"{entity_id} is not spatial")
        return entity['signals'].get(signal_name)
    
    def get_signal_at_position(self, entity_id, signal_name, position):
        field = self.get_signal_field(entity_id, signal_name)
        return field[tuple(position)] if field is not None else 0.0
    
    def set_signal_at_position(self, entity_id, signal_name, position, value):
        entity = self.entities[entity_id]
        if entity['representation'] == 'spatial':
            entity['signals'][signal_name][tuple(position)] = value
    
    # =========================================================================
    # AGENT-SPECIFIC
    # =========================================================================
    
    def add_agent(self, entity_id, agent_id, position=None, state=None):
        entity = self.entities[entity_id]
        if entity['representation'] != 'agents':
            raise ValueError(f"{entity_id} is not an agent population")
        
        agent = {
            'id': agent_id,
            'position': position,
            'state': state or {}
        }
        entity['agents'].append(agent)
        entity['count'] = len(entity['agents'])
    
    def get_agents(self, entity_id):
        entity = self.entities[entity_id]
        if entity['representation'] != 'agents':
            return []
        return entity['agents']
    
    def get_local_signal_for_agent(self, agent, signal_name):
        entity_id = agent.get('parent_entity')
        if not entity_id:
            return 0.0
        
        parent = self.entities.get(entity_id)
        if not parent:
            return 0.0
        
        if parent['representation'] == 'spatial':
            position = agent.get('position')
            if position:
                return self.get_signal_at_position(entity_id, signal_name, position)
        elif parent['representation'] == 'lumped':
            return parent['signals'].get(signal_name, 0.0)
        
        return 0.0
    
    # =========================================================================
    # SPATIAL QUERIES
    # =========================================================================
    
    def get_entity_position(self, entity_id):
        entity = self.entities.get(entity_id)
        if not entity:
            return None
        
        spatial_type = entity.get('spatial_type', 'localized')
        if spatial_type == 'distributed':
            return None
        
        return entity.get('position')
    
    def compute_distance(self, entity_id1, entity_id2):
        pos1 = self.get_entity_position(entity_id1)
        pos2 = self.get_entity_position(entity_id2)
        
        if pos1 is None or pos2 is None:
            return None
        
        return np.linalg.norm(np.array(pos1) - np.array(pos2))
    
    # =========================================================================
    # ORGANISM STATE
    # =========================================================================
    
    def set_organism_state(self, state_name, value):
        self.organism[state_name] = value
    
    def get_organism_state(self, state_name, default=None):
        return self.organism.get(state_name, default)
    
    # =========================================================================
    # ORGAN SYSTEM STATE
    # =========================================================================
    
    def set_system_state(self, system_name, state_name, value):
        """
        Set state for an organ system
        
        Args:
            system_name: Name of system (e.g., 'cardiovascular', 'respiratory')
            state_name: State variable name (e.g., 'total_peripheral_resistance')
            value: Value to set
        
        Example:
            state.set_system_state('cardiovascular', 'cardiac_output', 5.0)
            state.set_system_state('respiratory', 'minute_ventilation', 6.0)
        """
        if system_name not in self.organ_systems:
            self.organ_systems[system_name] = {}
        self.organ_systems[system_name][state_name] = value
    
    def get_system_state(self, system_name, state_name, default=None):
        """
        Get state from an organ system
        
        Args:
            system_name: Name of system
            state_name: State variable name
            default: Default value if not found
        
        Returns:
            State value or default
        """
        if system_name not in self.organ_systems:
            return default
        return self.organ_systems[system_name].get(state_name, default)
    
    def update_system_state(self, system_name, state_name, delta):
        """
        Update (increment/decrement) a system state
        
        Args:
            system_name: Name of system
            state_name: State variable name
            delta: Amount to add (can be negative)
        """
        current = self.get_system_state(system_name, state_name, 0.0)
        self.set_system_state(system_name, state_name, current + delta)
    
    def get_all_system_states(self, system_name):
        """
        Get all states for a system
        
        Returns:
            Dictionary of all state variables for the system
        """
        return self.organ_systems.get(system_name, {}).copy()
    
    # =========================================================================
    # HISTORY
    # =========================================================================
    
    def snapshot(self):
        snapshot = {
            'time': self.time,
            'entities': {},
            'organism': self.organism.copy(),
            'organ_systems': {}
        }
        
        # Copy organ system states
        for system_name, states in self.organ_systems.items():
            snapshot['organ_systems'][system_name] = states.copy()
        
        for entity_id, entity in self.entities.items():
            rep = entity['representation']
            
            if rep == 'lumped':
                snapshot['entities'][entity_id] = entity['signals'].copy()
            elif rep == 'spatial':
                snapshot['entities'][entity_id] = {}
                for signal_name, field in entity['signals'].items():
                    snapshot['entities'][entity_id][signal_name] = np.mean(field)
            elif rep == 'agents':
                snapshot['entities'][entity_id] = {
                    'count': entity['count']
                }
        
        return snapshot
    
    def record_history(self):
        self.history.append(self.snapshot())