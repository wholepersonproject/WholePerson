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
        
        # NEW: Constraint system
        self._constraints = {}  # (entity_id, signal_name) -> {min, max, warn_below, warn_above}
        self.enforce_constraints = True  # ← ADD THIS LINE (default: enabled)

    
    # =========================================================================
    # CONSTRAINT MANAGEMENT (NEW)
    # =========================================================================
    
    def add_constraint(self, entity_id, signal_name, **kwargs):
        """
        Add constraint to a signal
        
        Args:
            entity_id: 'blood', 'liver', 'organism', etc.
            signal_name: 'glucose', 'insulin', etc.
            **kwargs: Constraint parameters
                - min: Hard minimum (will clamp)
                - max: Hard maximum (will clamp)
                - warn_below: Warning threshold (low)
                - warn_above: Warning threshold (high)
                - unit: Unit of measurement (documentation)
        """
        key = (entity_id, signal_name)
        self._constraints[key] = kwargs
    
    def _apply_constraint(self, entity_id, signal_name, value):
        """
        Apply constraints to a value
        
        Returns:
            Constrained value (clamped if needed)
        """
        # ← ADD THIS CHECK AT THE TOP
        if not self.enforce_constraints:
            return value  # Skip all constraint logic

        key = (entity_id, signal_name)
        if key not in self._constraints:
            return value
        
        constraint = self._constraints[key]
        original_value = value
        
        # Clamp to hard bounds
        if 'min' in constraint and value < constraint['min']:
            value = constraint['min']
            if abs(original_value - value) > 0.01:  # Avoid spam for tiny violations
                print(f"  ⚠️  Clamped {entity_id}.{signal_name} = {original_value:.2f} → {value:.2f} (min: {constraint['min']})")
        
        if 'max' in constraint and value > constraint['max']:
            value = constraint['max']
            if abs(original_value - value) > 0.01:
                print(f"  ⚠️  Clamped {entity_id}.{signal_name} = {original_value:.2f} → {value:.2f} (max: {constraint['max']})")
        
        # Warn if crossing thresholds (but don't clamp)
        if 'warn_below' in constraint and value < constraint['warn_below']:
            if original_value >= constraint['warn_below']:  # Just crossed
                print(f"  ⚠️  WARNING: {entity_id}.{signal_name} = {value:.2f} below threshold {constraint['warn_below']}")
        
        if 'warn_above' in constraint and value > constraint['warn_above']:
            if original_value <= constraint['warn_above']:  # Just crossed
                print(f"  ⚠️  WARNING: {entity_id}.{signal_name} = {value:.2f} above threshold {constraint['warn_above']}")
        
        return value
    
    def get_constraint_info(self, entity_id, signal_name):
        """Get constraint metadata for a signal"""
        key = (entity_id, signal_name)
        return self._constraints.get(key)
    
    # =========================================================================
    # TISSUE MANAGEMENT
    # =========================================================================
    
    def add_tissue(self, tissue_id, tissue_type, **kwargs):
        """
        Add a tissue (collection of similar cells)
        STRUCTURAL CHANGE - triggers execution order update
        """
        self.tissues[tissue_id] = {
            'type': tissue_type,
            'signals': kwargs.get('signals', {}),
            'cell_types': kwargs.get('cell_types', []),
            **kwargs
        }
        
        if self._engine:
            self._engine.needs_reorder = True
    
    def remove_tissue(self, tissue_id, reason=""):
        """Remove a tissue"""
        if tissue_id not in self.tissues:
            return False
        
        del self.tissues[tissue_id]
        self._record_event('tissue_removed', f"Removed tissue '{tissue_id}': {reason}")
        
        if self._engine:
            self._engine.needs_reorder = True
        
        return True
    
    def get_tissue_state(self, tissue_id, state_name, default=None):
        """Get state from a tissue"""
        if tissue_id not in self.tissues:
            return default
        return self.tissues[tissue_id].get('signals', {}).get(state_name, default)
    
    def set_tissue_state(self, tissue_id, state_name, value):
        """Set state for a tissue (VALUE CHANGE)"""
        if tissue_id not in self.tissues:
            print(f"⚠️  Warning: Trying to set state '{state_name}' on non-existent tissue '{tissue_id}'")
            return
        
        if 'signals' not in self.tissues[tissue_id]:
            self.tissues[tissue_id]['signals'] = {}
        
        if state_name not in self.tissues[tissue_id]['signals']:
            print(f"ℹ️  Auto-creating signal '{state_name}' in tissue '{tissue_id}'")
        
        # NEW: Apply constraints
        value = self._apply_constraint(tissue_id, state_name, value)
        
        self.tissues[tissue_id]['signals'][state_name] = value
    
    # =========================================================================
    # ORGAN MANAGEMENT
    # =========================================================================
    
    def add_organ(self, organ_id, organ_type, **kwargs):
        """Add an organ (STRUCTURAL CHANGE)"""
        self.organs[organ_id] = {
            'type': organ_type,
            'signals': {},
            'tissues': kwargs.get('tissues', []),
            'mass': kwargs.get('mass'),
            **kwargs
        }
        
        representation = kwargs.get('representation', 'lumped')
        
        if representation == 'spatial':
            shape = tuple(kwargs['shape'])
            self.organs[organ_id]['shape'] = shape
            self.organs[organ_id]['dx'] = kwargs.get('dx', 1.0)
            # Initialize spatial fields as ndarrays
            for signal_name, initial_value in kwargs.get('signals', {}).items():
                if isinstance(initial_value, np.ndarray):
                    self.organs[organ_id]['signals'][signal_name] = initial_value
                else:
                    self.organs[organ_id]['signals'][signal_name] = \
                        np.full(shape, initial_value, dtype=np.float32)
        else:
            self.organs[organ_id]['signals'] = kwargs.get('signals', {})
        
        self._record_event('organ_added', f"Added organ '{organ_id}'")
        
        if self._engine:
            self._engine.needs_reorder = True
    
    def remove_organ(self, organ_id, reason=""):
        """Remove an organ"""
        if organ_id not in self.organs:
            return False
        
        del self.organs[organ_id]
        self._record_event('organ_removed', f"Removed organ '{organ_id}': {reason}")
        
        if self._engine:
            self._engine.needs_reorder = True
        
        return True
    
    def get_organ_state(self, organ_id, state_name, default=None):
        """Get state from an organ"""
        if organ_id not in self.organs:
            return default
        return self.organs[organ_id].get('signals', {}).get(state_name, default)
    
    def set_organ_state(self, organ_id, state_name, value):
        """Set state for an organ (VALUE CHANGE)"""
        if organ_id not in self.organs:
            print(f"⚠️  Warning: Trying to set state '{state_name}' on non-existent organ '{organ_id}'")
            return
        
        if 'signals' not in self.organs[organ_id]:
            self.organs[organ_id]['signals'] = {}
        
        if state_name not in self.organs[organ_id]['signals']:
            print(f"ℹ️  Auto-creating signal '{state_name}' in organ '{organ_id}'")
        
        # NEW: Apply constraints
        value = self._apply_constraint(organ_id, state_name, value)
        
        self.organs[organ_id]['signals'][state_name] = value
    
    # =========================================================================
    # ENTITY MANAGEMENT
    # =========================================================================
    
    def add_entity(self, entity_id, entity_type, representation, **kwargs):
        """Add an entity (STRUCTURAL CHANGE)"""
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
        
        if self._engine:
            self._engine.needs_reorder = True
    
    def remove_entity(self, entity_id, reason=""):
        """Remove an entity (STRUCTURAL CHANGE)"""
        if entity_id not in self.entities:
            return False
        
        del self.entities[entity_id]
        self._record_event('entity_removed', f"Removed entity '{entity_id}': {reason}")
        
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
    
    def remove_flow(self, flow_id, reason=""):
        """Remove a flow"""
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
        
        # Try organs second
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
        
        # Try organism fourth
        if id == 'organism':
            return self.organism.get(signal_name, 0.0)
        
        return None
    
    def update_signal(self, id, signal_name, delta):
        """Update signal value by delta with smart lookup"""
        # Try entities first
        if id in self.entities:
            entity = self.entities[id]
            rep = entity['representation']
            
            if rep == 'lumped':
                current = entity['signals'].get(signal_name, 0.0)
                if signal_name not in entity['signals']:
                    print(f"ℹ️  Auto-creating signal '{signal_name}' in entity '{id}'")
                # NEW: Apply constraints
                new_value = self._apply_constraint(id, signal_name, current + delta)
                entity['signals'][signal_name] = new_value
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
        
        # Try organs second
        if id in self.organs:
            organ = self.organs[id]
            rep = organ.get('representation', 'lumped')
            
            if rep == 'lumped':
                if 'signals' not in organ:
                    organ['signals'] = {}
                current = organ['signals'].get(signal_name, 0.0)
                if signal_name not in organ['signals']:
                    print(f"ℹ️  Auto-creating signal '{signal_name}' in organ '{id}'")
                # NEW: Apply constraints
                new_value = self._apply_constraint(id, signal_name, current + delta)
                organ['signals'][signal_name] = new_value
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
            # NEW: Apply constraints
            new_value = self._apply_constraint(id, signal_name, current + delta)
            self.tissues[id]['signals'][signal_name] = new_value
            return
        
        # Try organism fourth
        if id == 'organism':
            current = self.organism.get(signal_name, 0.0)
            new_value = self._apply_constraint(id, signal_name, current + delta)
            self.organism[signal_name] = new_value
            return
        
        print(f"⚠️  Warning: Cannot update signal '{signal_name}' - '{id}' not found")
    
    def set_signal(self, id, signal_name, value):
        """Set signal value with smart lookup"""
        # NEW: Apply constraints
        value = self._apply_constraint(id, signal_name, value)
        
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
        
        # Try organs second
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
        
        # Try organism fourth
        if id == 'organism':
            self.organism[signal_name] = self._apply_constraint(id, signal_name, value)
            return
        
        print(f"⚠️  Warning: Cannot set signal '{signal_name}' - '{id}' not found")
    
    # =========================================================================
    # SPATIAL-SPECIFIC
    # =========================================================================
    
    def get_field(self, id, signal_name):
        """
        Get the raw spatial field (ndarray) for a signal.
        Searches: organs → entities
        
        Unlike get_signal() which returns the mean of a spatial field,
        this returns the full ndarray for direct spatial manipulation
        (e.g. diffusion, local reads by spatially-positioned agents).
        
        Returns:
            np.ndarray if spatial, None otherwise
        """
        # Try organs first (bone, kidney, etc.)
        if id in self.organs:
            organ = self.organs[id]
            if organ.get('representation') == 'spatial':
                return organ.get('signals', {}).get(signal_name)
        
        # Try entities second
        if id in self.entities:
            entity = self.entities[id]
            if entity['representation'] == 'spatial':
                return entity['signals'].get(signal_name)
        
        return None
    
    def set_field(self, id, signal_name, field):
        """
        Set the raw spatial field (ndarray) for a signal.
        Searches: organs → entities
        
        Replaces the entire field array. For in-place modifications,
        get_field() and modify the returned array directly.
        
        Args:
            id: organ or entity identifier
            signal_name: signal name
            field: np.ndarray matching the organ/entity shape
        """
        # Try organs first
        if id in self.organs:
            organ = self.organs[id]
            if organ.get('representation') == 'spatial':
                if signal_name in organ.get('signals', {}):
                    organ['signals'][signal_name] = field
                else:
                    print(f"⚠️  Warning: Signal '{signal_name}' doesn't exist in spatial organ '{id}'")
                return
        
        # Try entities second
        if id in self.entities:
            entity = self.entities[id]
            if entity['representation'] == 'spatial':
                if signal_name in entity['signals']:
                    entity['signals'][signal_name] = field
                else:
                    print(f"⚠️  Warning: Signal '{signal_name}' doesn't exist in spatial entity '{id}'")
                return
        
        print(f"⚠️  Warning: Cannot set field '{signal_name}' - '{id}' not found or not spatial")

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
    # GEOMETRY QUERIES
    # =========================================================================
    
    def get_organ_geometry(self, organ_id):
        """Get geometry description for an organ"""
        organ = self.organs.get(organ_id)
        if not organ:
            return None
        return organ.get('geometry')
    
    def get_organ_bounding_box(self, organ_id):
        """
        Compute axis-aligned bounding box from geometry primitives
        
        Returns:
            dict with 'min' and 'max' arrays in world coordinates,
            or None if no geometry defined
        """
        geo = self.get_organ_geometry(organ_id)
        if not geo or 'primitives' not in geo:
            return None
        
        mins = np.array([np.inf, np.inf, np.inf])
        maxs = np.array([-np.inf, -np.inf, -np.inf])
        
        for prim in geo['primitives']:
            offset = np.array(prim.get('offset', [0, 0, 0]), dtype=float)
            if prim['shape'] == 'sphere':
                r = prim['radius']
                mins = np.minimum(mins, offset - r)
                maxs = np.maximum(maxs, offset + r)
            elif prim['shape'] == 'ellipsoid':
                radii = np.array(prim['radii'], dtype=float)
                mins = np.minimum(mins, offset - radii)
                maxs = np.maximum(maxs, offset + radii)
            elif prim['shape'] == 'box':
                half = np.array(prim['dimensions'], dtype=float) / 2
                mins = np.minimum(mins, offset - half)
                maxs = np.maximum(maxs, offset + half)
        
        anchor = np.array(self.organs[organ_id].get('position', [0, 0, 0]), dtype=float)
        return {'min': anchor + mins, 'max': anchor + maxs}
    
    def point_in_organ(self, organ_id, point):
        """
        Test whether a world-space point falls inside any geometry primitive
        
        Args:
            organ_id: Organ identifier
            point: [x, y, z] in world coordinates
            
        Returns:
            True/False, or None if no geometry defined
        """
        geo = self.get_organ_geometry(organ_id)
        if not geo or 'primitives' not in geo:
            return None
        
        anchor = np.array(self.organs[organ_id].get('position', [0, 0, 0]), dtype=float)
        local_pt = np.array(point, dtype=float) - anchor
        
        for prim in geo['primitives']:
            offset = np.array(prim.get('offset', [0, 0, 0]), dtype=float)
            rel = local_pt - offset
            
            if prim['shape'] == 'sphere':
                if np.linalg.norm(rel) <= prim['radius']:
                    return True
            elif prim['shape'] == 'ellipsoid':
                radii = np.array(prim['radii'], dtype=float)
                if np.sum((rel / radii) ** 2) <= 1.0:
                    return True
            elif prim['shape'] == 'box':
                half = np.array(prim['dimensions'], dtype=float) / 2
                if np.all(np.abs(rel) <= half):
                    return True
        
        return False
    
    # =========================================================================
    # ORGANISM STATE
    # =========================================================================
    
    def set_organism_state(self, state_name, value):
        """Set organism-level state with constraints"""
        # NEW: Apply constraints
        value = self._apply_constraint('organism', state_name, value)
        self.organism[state_name] = value
    
    def get_organism_state(self, state_name, default=None):
        """Get organism-level state"""
        return self.organism.get(state_name, default)
    
    # =========================================================================
    # ORGAN SYSTEM STATE
    # =========================================================================
    
    def set_system_state(self, system_name, state_name, value):
        """Set state for an organ system"""
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
    # STATE VALIDATION
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
        
        # Copy organ states
        for organ_id, organ in self.organs.items():
            rep = organ.get('representation', 'lumped')
            
            if rep == 'lumped':
                snapshot['organs'][organ_id] = organ.get('signals', {}).copy()
            elif rep == 'spatial':
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