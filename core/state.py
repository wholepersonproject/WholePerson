import numpy as np

class SimulationState:
    '''
    Unified state with hierarchical organization
    
    Hierarchy levels:
      - molecules/cells (entities)
      - tissues
      - organs
      - organ_systems
      - organism
    '''
    
    def __init__(self):
        self.time = 0.0
        
        # Hierarchical organization (from small to large)
        self.entities = {}       # Molecules, cells, cell populations
        self.tissues = {}        # Collections of similar cells
        self.organs = {}         # Collections of tissues
        self.organ_systems = {}  # Collections of organs
        self.organism = {}       # Whole-body state
        
        self.flows = {}
        self.history = []
        self._engine = None  # Set by engine for dynamic changes
        self.events = []     # Track major state changes
    
    # =========================================================================
    # TISSUE MANAGEMENT
    # =========================================================================
    
    def add_tissue(self, tissue_id, tissue_type, **kwargs):
        """
        Add a tissue (collection of similar cells)
        STRUCTURAL CHANGE - triggers execution order update
        
        Args:
            tissue_id: Unique identifier (e.g., 'cardiac_muscle')
            tissue_type: Type of tissue (e.g., 'muscle', 'epithelial', 'connective')
            **kwargs: Additional properties (signals, cell_types, etc.)
        """
        self.tissues[tissue_id] = {
            'type': tissue_type,
            'signals': kwargs.get('signals', {}),
            'cell_types': kwargs.get('cell_types', []),
            **kwargs
        }
        
        # Trigger engine update (STRUCTURAL change)
        if self._engine:
            self._engine.needs_reorder = True
    
    def remove_tissue(self, tissue_id, reason=""):
        """
        Remove a tissue
        STRUCTURAL CHANGE - triggers execution order update
        """
        if tissue_id not in self.tissues:
            return False
        
        del self.tissues[tissue_id]
        self._record_event('tissue_removed', f"Removed tissue '{tissue_id}': {reason}")
        
        # Trigger engine update (STRUCTURAL change)
        if self._engine:
            self._engine.needs_reorder = True
        
        return True
    
    def get_tissue_state(self, tissue_id, state_name, default=None):
        """Get state from a tissue"""
        if tissue_id not in self.tissues:
            return default
        return self.tissues[tissue_id].get('signals', {}).get(state_name, default)
    
    def set_tissue_state(self, tissue_id, state_name, value):
        """
        Set state for a tissue
        VALUE CHANGE - does NOT trigger reorder
        """
        if tissue_id not in self.tissues:
            print(f"⚠️  Warning: Trying to set state '{state_name}' on non-existent tissue '{tissue_id}'")
            return
        
        if 'signals' not in self.tissues[tissue_id]:
            self.tissues[tissue_id]['signals'] = {}
        
        if state_name not in self.tissues[tissue_id]['signals']:
            print(f"ℹ️  Auto-creating signal '{state_name}' in tissue '{tissue_id}'")
        
        self.tissues[tissue_id]['signals'][state_name] = value
        # No reorder - just a value change
    
    # =========================================================================
    # ORGAN MANAGEMENT
    # =========================================================================
    
    def add_organ(self, organ_id, organ_type, **kwargs):
        """
        Add an organ (collection of tissues)
        STRUCTURAL CHANGE - triggers execution order update
        
        Args:
            organ_id: Unique identifier (e.g., 'heart', 'liver', 'kidney')
            organ_type: Type of organ
            **kwargs: Additional properties (signals, tissues, mass, etc.)
        """
        self.organs[organ_id] = {
            'type': organ_type,
            'signals': kwargs.get('signals', {}),
            'tissues': kwargs.get('tissues', []),
            'mass': kwargs.get('mass'),
            **kwargs
        }
        
        self._record_event('organ_added', f"Added organ '{organ_id}'")
        
        # Trigger engine update (STRUCTURAL change)
        if self._engine:
            self._engine.needs_reorder = True
    
    def remove_organ(self, organ_id, reason=""):
        """
        Remove an organ (failure, surgical removal)
        STRUCTURAL CHANGE - triggers execution order update
        """
        if organ_id not in self.organs:
            return False
        
        del self.organs[organ_id]
        self._record_event('organ_removed', f"Removed organ '{organ_id}': {reason}")
        
        # Trigger engine update (STRUCTURAL change)
        if self._engine:
            self._engine.needs_reorder = True
        
        return True
    
    def get_organ_state(self, organ_id, state_name, default=None):
        """Get state from an organ"""
        if organ_id not in self.organs:
            return default
        return self.organs[organ_id].get('signals', {}).get(state_name, default)
    
    def set_organ_state(self, organ_id, state_name, value):
        """
        Set state for an organ
        VALUE CHANGE - does NOT trigger reorder
        """
        if organ_id not in self.organs:
            print(f"⚠️  Warning: Trying to set state '{state_name}' on non-existent organ '{organ_id}'")
            return
        
        if 'signals' not in self.organs[organ_id]:
            self.organs[organ_id]['signals'] = {}
        
        if state_name not in self.organs[organ_id]['signals']:
            print(f"ℹ️  Auto-creating signal '{state_name}' in organ '{organ_id}'")
        
        self.organs[organ_id]['signals'][state_name] = value
        # No reorder - just a value change
    
    # =========================================================================
    # ENTITY MANAGEMENT
    # =========================================================================
    
    def add_entity(self, entity_id, entity_type, representation, **kwargs):
        """
        Add an entity
        STRUCTURAL CHANGE - triggers execution order update
        """
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
        
        # Trigger engine update (STRUCTURAL change)
        if self._engine:
            self._engine.needs_reorder = True
    
    def remove_entity(self, entity_id, reason=""):
        """
        Remove an entity (amputation, cell death, tissue loss)
        STRUCTURAL CHANGE - triggers execution order update
        """
        if entity_id not in self.entities:
            return False
        
        del self.entities[entity_id]
        self._record_event('entity_removed', f"Removed entity '{entity_id}': {reason}")
        
        # Trigger engine update (STRUCTURAL change)
        if self._engine:
            self._engine.needs_reorder = True
        
        return True
    
    def add_flow(self, flow_id, from_entity, to_entity, rate, flow_type='transport'):
        """Add a flow between entities"""
        self.flows[flow_id] = {
            'from': from_entity,
            'to': to_entity,
            'rate': rate,
            'type': flow_type
        }
        # Flows don't typically affect process execution, so no reorder
    
    def remove_flow(self, flow_id, reason=""):
        """Remove a flow (vascular occlusion, etc.)"""
        if flow_id not in self.flows:
            return False
        
        del self.flows[flow_id]
        self._record_event('flow_removed', f"Removed flow '{flow_id}': {reason}")
        
        return True
    
    # =========================================================================
    # SIGNAL ACCESS (unified API for all representations)
    # =========================================================================
    
    def get_signal(self, id, signal_name):
        """
        Get signal value with smart lookup
        Searches: entities → organs → tissues
        Handles lumped and spatial representations
        READ operation - does NOT trigger reorder
        """
        # Try entities first
        if id in self.entities:
            entity = self.entities[id]
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
        
        # Try organs second (can also be spatial!)
        if id in self.organs:
            organ = self.organs[id]
            rep = organ.get('representation', 'lumped')
            
            if rep == 'lumped':
                return organ.get('signals', {}).get(signal_name, 0.0)
            elif rep == 'spatial':
                field = organ.get('signals', {}).get(signal_name)
                return np.mean(field) if field is not None else 0.0
            return 0.0
        
        # Try tissues third
        if id in self.tissues:
            return self.tissues[id].get('signals', {}).get(signal_name, 0.0)
        
        # Not found anywhere
        return None
    
    def update_signal(self, id, signal_name, delta):
        """
        Update signal value by delta with smart lookup
        Searches: entities → organs → tissues
        Handles lumped and spatial representations
        VALUE CHANGE - does NOT trigger reorder
        """
        # Try entities first
        if id in self.entities:
            entity = self.entities[id]
            rep = entity['representation']
            
            if rep == 'lumped':
                current = entity['signals'].get(signal_name, 0.0)
                if signal_name not in entity['signals']:
                    print(f"ℹ️  Auto-creating signal '{signal_name}' in entity '{id}'")
                entity['signals'][signal_name] = current + delta
            elif rep == 'spatial':
                if signal_name in entity['signals']:
                    entity['signals'][signal_name] += delta
                else:
                    print(f"⚠️  Warning: Signal '{signal_name}' doesn't exist in spatial entity '{id}'")
            elif rep == 'agents':
                for agent in entity['agents']:
                    current = agent['state'].get(signal_name, 0.0)
                    agent['state'][signal_name] = current + delta
            return
        
        # Try organs second (can also be spatial!)
        if id in self.organs:
            organ = self.organs[id]
            rep = organ.get('representation', 'lumped')
            
            if rep == 'lumped':
                if 'signals' not in organ:
                    organ['signals'] = {}
                current = organ['signals'].get(signal_name, 0.0)
                if signal_name not in organ['signals']:
                    print(f"ℹ️  Auto-creating signal '{signal_name}' in organ '{id}'")
                organ['signals'][signal_name] = current + delta
            elif rep == 'spatial':
                if 'signals' not in organ:
                    organ['signals'] = {}
                if signal_name in organ['signals']:
                    organ['signals'][signal_name] += delta
                else:
                    print(f"⚠️  Warning: Signal '{signal_name}' doesn't exist in spatial organ '{id}'")
            return
        
        # Try tissues third
        if id in self.tissues:
            if 'signals' not in self.tissues[id]:
                self.tissues[id]['signals'] = {}
            current = self.tissues[id]['signals'].get(signal_name, 0.0)
            if signal_name not in self.tissues[id]['signals']:
                print(f"ℹ️  Auto-creating signal '{signal_name}' in tissue '{id}'")
            self.tissues[id]['signals'][signal_name] = current + delta
            return
        
        # Not found anywhere
        print(f"⚠️  Warning: Cannot update signal '{signal_name}' - '{id}' not found")
    
    def set_signal(self, id, signal_name, value):
        """
        Set signal value with smart lookup
        Searches: entities → organs → tissues
        Handles lumped and spatial representations
        VALUE CHANGE - does NOT trigger reorder
        """
        # Try entities first
        if id in self.entities:
            entity = self.entities[id]
            rep = entity['representation']
            
            if rep == 'lumped':
                if signal_name not in entity['signals']:
                    print(f"ℹ️  Auto-creating signal '{signal_name}' in entity '{id}'")
                entity['signals'][signal_name] = value
            elif rep == 'spatial':
                if signal_name in entity['signals']:
                    entity['signals'][signal_name][:] = value
                else:
                    print(f"⚠️  Warning: Signal '{signal_name}' doesn't exist in spatial entity '{id}'")
            elif rep == 'agents':
                for agent in entity['agents']:
                    agent['state'][signal_name] = value
            return
        
        # Try organs second (can also be spatial!)
        if id in self.organs:
            organ = self.organs[id]
            rep = organ.get('representation', 'lumped')
            
            if rep == 'lumped':
                if 'signals' not in organ:
                    organ['signals'] = {}
                if signal_name not in organ['signals']:
                    print(f"ℹ️  Auto-creating signal '{signal_name}' in organ '{id}'")
                organ['signals'][signal_name] = value
            elif rep == 'spatial':
                if 'signals' not in organ:
                    organ['signals'] = {}
                if signal_name in organ['signals']:
                    organ['signals'][signal_name][:] = value
                else:
                    print(f"⚠️  Warning: Signal '{signal_name}' doesn't exist in spatial organ '{id}'")
            return
        
        # Try tissues third
        if id in self.tissues:
            if 'signals' not in self.tissues[id]:
                self.tissues[id]['signals'] = {}
            if signal_name not in self.tissues[id]['signals']:
                print(f"ℹ️  Auto-creating signal '{signal_name}' in tissue '{id}'")
            self.tissues[id]['signals'][signal_name] = value
            return
        
        # Not found anywhere
        print(f"⚠️  Warning: Cannot set signal '{signal_name}' - '{id}' not found")
    
    # =========================================================================
    # SPATIAL-SPECIFIC
    # =========================================================================
    
    def get_signal_field(self, entity_id, signal_name):
        entity = self.entities.get(entity_id)
        if not entity or entity['representation'] != 'spatial':
            return None
        return entity['signals'].get(signal_name)
    
    def get_signal_at_position(self, entity_id, signal_name, position):
        field = self.get_signal_field(entity_id, signal_name)
        return field[tuple(position)] if field is not None else 0.0
    
    def set_signal_at_position(self, entity_id, signal_name, position, value):
        entity = self.entities.get(entity_id)
        if entity and entity['representation'] == 'spatial':
            if signal_name in entity['signals']:
                entity['signals'][signal_name][tuple(position)] = value
    
    # =========================================================================
    # AGENT-SPECIFIC
    # =========================================================================
    
    def add_agent(self, entity_id, agent_id, position=None, state=None):
        entity = self.entities.get(entity_id)
        if not entity or entity['representation'] != 'agents':
            return
        
        agent = {
            'id': agent_id,
            'position': position,
            'state': state or {}
        }
        entity['agents'].append(agent)
        entity['count'] = len(entity['agents'])
    
    def get_agents(self, entity_id):
        entity = self.entities.get(entity_id)
        if not entity or entity['representation'] != 'agents':
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
        """Set organism-level state (VALUE change, no reorder)"""
        self.organism[state_name] = value
    
    def get_organism_state(self, state_name, default=None):
        """Get organism-level state"""
        return self.organism.get(state_name, default)
    
    # =========================================================================
    # ORGAN SYSTEM STATE
    # =========================================================================
    
    def set_system_state(self, system_name, state_name, value):
        """Set state for an organ system (VALUE change, no reorder)"""
        if system_name not in self.organ_systems:
            self.organ_systems[system_name] = {}
        self.organ_systems[system_name][state_name] = value
    
    def get_system_state(self, system_name, state_name, default=None):
        """Get state from an organ system"""
        if system_name not in self.organ_systems:
            return default
        return self.organ_systems[system_name].get(state_name, default)
    
    def update_system_state(self, system_name, state_name, delta):
        """Update (increment/decrement) a system state"""
        current = self.get_system_state(system_name, state_name, 0.0)
        self.set_system_state(system_name, state_name, current + delta)
    
    def get_all_system_states(self, system_name):
        """Get all states for a system"""
        return self.organ_systems.get(system_name, {}).copy()
    
    # =========================================================================
    # EVENT LOGGING
    # =========================================================================
    
    def _record_event(self, event_type, description):
        """Internal event recording"""
        event = {
            'time': self.time,
            'type': event_type,
            'description': description
        }
        self.events.append(event)
        print(f"⚠️  EVENT at t={self.time/3600:.1f}h: {description}")
    
    def get_events(self, event_type=None):
        """Get logged events, optionally filtered by type"""
        if event_type:
            return [e for e in self.events if e['type'] == event_type]
        return self.events.copy()
    
    # =========================================================================
    # STATE VALIDATION (checks if inputs/outputs exist)
    # =========================================================================
    
    def has_entity_signal(self, entity_id, signal_name):
        """Check if entity and signal exist"""
        entity = self.entities.get(entity_id)
        if not entity:
            return False
        
        rep = entity['representation']
        if rep in ['lumped', 'spatial']:
            return signal_name in entity.get('signals', {})
        elif rep == 'agents':
            # For agents, check if at least one agent has the signal
            agents = entity.get('agents', [])
            return len(agents) > 0 and any(signal_name in a.get('state', {}) for a in agents)
        
        return False
    
    def has_organ_signal(self, organ_id, signal_name):
        """Check if organ and signal exist"""
        organ = self.organs.get(organ_id)
        if not organ:
            return False
        return signal_name in organ.get('signals', {})
    
    def has_tissue_signal(self, tissue_id, signal_name):
        """Check if tissue and signal exist"""
        tissue = self.tissues.get(tissue_id)
        if not tissue:
            return False
        return signal_name in tissue.get('signals', {})
    
    def has_organism_state(self, state_name):
        """Check if organism state exists"""
        return state_name in self.organism
    
    # =========================================================================
    # HISTORY
    # =========================================================================
    
    def snapshot(self):
        snapshot = {
            'time': self.time,
            'entities': {},
            'tissues': {},
            'organs': {},
            'organ_systems': {},
            'organism': self.organism.copy()
        }
        
        # Copy tissue states
        for tissue_id, tissue in self.tissues.items():
            snapshot['tissues'][tissue_id] = tissue.get('signals', {}).copy()
        
        # Copy organ states (handle spatial organs)
        for organ_id, organ in self.organs.items():
            rep = organ.get('representation', 'lumped')
            
            if rep == 'lumped':
                snapshot['organs'][organ_id] = organ.get('signals', {}).copy()
            elif rep == 'spatial':
                # Average spatial fields for history
                snapshot['organs'][organ_id] = {}
                for signal_name, field in organ.get('signals', {}).items():
                    if isinstance(field, np.ndarray):
                        snapshot['organs'][organ_id][signal_name] = np.mean(field)
                    else:
                        snapshot['organs'][organ_id][signal_name] = field
        
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