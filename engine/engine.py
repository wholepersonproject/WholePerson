from core.graph import DependencyGraph
import pickle
import warnings

class PhysiologyEngine:
    def __init__(self, state):
        self.state = state
        self.state._engine = self  # Allow dynamic changes to access engine
        self.models = {}
        self.graph = DependencyGraph()
        self.execution_order = []
        self.last_run_time = {}
        self.perturbation_manager = None
        self._needs_reorder = False  # Track if we need to recompute execution order
        self._deletion_queue = []  # General deletion queue
        
        # NEW: Events system
        self._events = {}  # event_name -> (condition, action)
        
        # NEW: Validation system
        self._constraints = {}  # (entity_id, signal_name) -> (min, max, clamp, warn)
        
        # NEW: Checkpointing
        self._checkpoint_counter = 0
    
    def register_model(self, process_id, model, dependencies=None):
        self.models[process_id] = model
        self.graph.add_process(process_id)
        self.last_run_time[process_id] = -model.timescale.value
        
        if dependencies:
            for dep in dependencies:
                self.graph.add_dependency(dep, process_id)
        
        self._needs_reorder = True  # Need to recompute order
    
    # =========================================================================
    # DYNAMIC PROCESS MANAGEMENT (NEW!)
    # =========================================================================
    
    def add_process(self, process_id, model, dependencies=None):
        """
        Add a process during runtime
        
        Args:
            process_id: Unique identifier
            model: ProcessModel instance
            dependencies: List of process_ids this depends on
        """
        self.register_model(process_id, model, dependencies)
        print(f"✓ Added process: {process_id}")
    
    def remove_process(self, process_id):
        """
        Remove a process during runtime
        
        Args:
            process_id: Process to remove
        """
        if process_id in self.models:
            del self.models[process_id]
            self.graph.remove_process(process_id)
            if process_id in self.last_run_time:
                del self.last_run_time[process_id]
            self._needs_reorder = True
            print(f"✓ Removed process: {process_id}")
    
    def has_process(self, process_id):
        """Check if a process exists"""
        return process_id in self.models
    
    def get_process(self, process_id):
        """Get a process by ID"""
        return self.models.get(process_id)
    
    def list_processes(self):
        """List all registered processes"""
        return list(self.models.keys())
    
    # =========================================================================
    # GENERAL DELETION SYSTEM
    # =========================================================================
    
    def mark_for_deletion(self, item_type, *args):
        """
        Mark anything for deletion at end of timestep
        
        UNIVERSAL deletion method - works for everything!
        
        Usage:
            # Agents
            engine.mark_for_deletion('agent', 'cells', 'cell_1')
            
            # Entities
            engine.mark_for_deletion('entity', 'old_protein')
            
            # Tissues
            engine.mark_for_deletion('tissue', 'damaged_tissue')
            
            # Organs
            engine.mark_for_deletion('organ', 'failed_kidney')
            
            # Processes
            engine.mark_for_deletion('process', 'process_id')
        
        Everything gets deleted at END of timestep automatically.
        No crashes, ever.
        """
        self._deletion_queue.append((item_type, args))
    
    def _apply_deletions(self):
        """
        Apply all pending deletions
        
        Called automatically at end of each timestep
        """
        if not self._deletion_queue:
            return
        
        for item_type, args in self._deletion_queue:
            if item_type == 'agent':
                # args = (population_id, agent_id)
                population_id, agent_id = args
                if population_id in self.state.entities:
                    agents = self.state.entities[population_id]['agents']
                    agent = next((a for a in agents if a['id'] == agent_id), None)
                    if agent:
                        agents.remove(agent)
                        self.state.entities[population_id]['count'] = len(agents)
            
            elif item_type == 'entity':
                # args = (entity_id,)
                entity_id = args[0]
                if entity_id in self.state.entities:
                    del self.state.entities[entity_id]
            
            elif item_type == 'tissue':
                # args = (tissue_id,)
                tissue_id = args[0]
                if tissue_id in self.state.tissues:
                    del self.state.tissues[tissue_id]
            
            elif item_type == 'organ':
                # args = (organ_id,)
                organ_id = args[0]
                if organ_id in self.state.organs:
                    del self.state.organs[organ_id]
            
            elif item_type == 'process':
                # args = (process_id,)
                process_id = args[0]
                self.remove_process(process_id)
        
        # Clear queue
        self._deletion_queue.clear()
    
    # =========================================================================
    # EVENTS & CALLBACKS SYSTEM
    # =========================================================================
    
    def add_event(self, event_name, condition, action):
        """
        Add an event that triggers when condition is met
        
        Args:
            event_name: Unique identifier for this event
            condition: Function that takes (state) and returns bool
            action: Function that takes (state, engine) and performs action
        
        Example:
            engine.add_event('hypoglycemia',
                condition=lambda s: s.get_signal('blood', 'glucose') < 40,
                action=lambda s, e: trigger_counter_regulation(s, e))
        """
        self._events[event_name] = (condition, action)
    
    def remove_event(self, event_name):
        """Remove an event"""
        if event_name in self._events:
            del self._events[event_name]
    
    def has_event(self, event_name):
        """Check if event exists"""
        return event_name in self._events
    
    def list_events(self):
        """List all registered events"""
        return list(self._events.keys())
    
    def _check_events(self):
        """
        Check all events and trigger actions if conditions met
        
        Called automatically after each timestep
        """
        for event_name, (condition, action) in self._events.items():
            try:
                if condition(self.state):
                    action(self.state, self)
            except Exception as e:
                warnings.warn(f"Event '{event_name}' failed: {e}")
    
    # =========================================================================
    # VALIDATION & CONSTRAINTS SYSTEM
    # =========================================================================
    
    def add_constraint(self, entity_id, signal_name, min_val=None, max_val=None, 
                       clamp=False, warn=True):
        """
        Add validation constraint for a signal
        
        Args:
            entity_id: Entity to constrain
            signal_name: Signal to constrain
            min_val: Minimum allowed value (None = no minimum)
            max_val: Maximum allowed value (None = no maximum)
            clamp: If True, clamp values to bounds. If False, just warn.
            warn: If True, print warnings when bounds violated
        
        Example:
            engine.add_constraint('blood', 'glucose', min_val=0, max_val=500, 
                                 clamp=True, warn=True)
        """
        self._constraints[(entity_id, signal_name)] = {
            'min': min_val,
            'max': max_val,
            'clamp': clamp,
            'warn': warn
        }
    
    def remove_constraint(self, entity_id, signal_name):
        """Remove a constraint"""
        key = (entity_id, signal_name)
        if key in self._constraints:
            del self._constraints[key]
    
    def list_constraints(self):
        """List all constraints"""
        return list(self._constraints.keys())
    
    def _validate_state(self):
        """
        Validate all state values against constraints
        
        Called automatically after each timestep
        """
        for (entity_id, signal_name), constraint in self._constraints.items():
            # Check if entity exists
            if entity_id not in self.state.entities:
                continue
            
            # Get current value
            try:
                value = self.state.get_signal(entity_id, signal_name)
            except:
                continue
            
            min_val = constraint['min']
            max_val = constraint['max']
            clamp = constraint['clamp']
            warn = constraint['warn']
            
            # Check bounds
            violated = False
            new_value = value
            
            if min_val is not None and value < min_val:
                violated = True
                if clamp:
                    new_value = min_val
                if warn:
                    warnings.warn(
                        f"Constraint violation: {entity_id}.{signal_name} = {value:.3f} "
                        f"< min({min_val})" + 
                        (f" [clamped to {min_val}]" if clamp else "")
                    )
            
            if max_val is not None and value > max_val:
                violated = True
                if clamp:
                    new_value = max_val
                if warn:
                    warnings.warn(
                        f"Constraint violation: {entity_id}.{signal_name} = {value:.3f} "
                        f"> max({max_val})" +
                        (f" [clamped to {max_val}]" if clamp else "")
                    )
            
            # Apply clamping if needed
            if violated and clamp:
                self.state.set_signal(entity_id, signal_name, new_value)
    
    # =========================================================================
    # CHECKPOINTING SYSTEM
    # =========================================================================
    
    def save_checkpoint(self, filename=None):
        """
        Save complete simulation state to file
        
        Args:
            filename: Path to save checkpoint (default: auto-generated)
        
        Returns:
            filename: Path where checkpoint was saved
        
        Example:
            engine.save_checkpoint('simulation_t1000.pkl')
        
        Note: Events with lambda functions cannot be saved.
              Register events again after loading checkpoint.
        """
        if filename is None:
            filename = f"checkpoint_{self._checkpoint_counter:04d}.pkl"
            self._checkpoint_counter += 1
        
        # Temporarily disconnect engine from state (can't pickle back-reference)
        temp_engine = self.state._engine
        self.state._engine = None
        
        checkpoint = {
            'state': self.state,
            'models': self.models,
            'execution_order': self.execution_order,
            'last_run_time': self.last_run_time,
            'constraints': self._constraints,
            'checkpoint_counter': self._checkpoint_counter
        }
        
        # Note: We skip events because they may contain unpicklable lambdas
        # User should re-register events after loading
        
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Reconnect engine
        self.state._engine = temp_engine
        
        return filename
    
    def load_checkpoint(self, filename):
        """
        Load simulation state from checkpoint file
        
        Args:
            filename: Path to checkpoint file
        
        Example:
            engine.load_checkpoint('simulation_t1000.pkl')
        
        Note: Events are not saved in checkpoints.
              Re-register events after loading if needed.
        """
        with open(filename, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.state = checkpoint['state']
        self.state._engine = self  # Reconnect engine
        self.models = checkpoint['models']
        self.execution_order = checkpoint['execution_order']
        self.last_run_time = checkpoint['last_run_time']
        self._constraints = checkpoint.get('constraints', {})
        self._checkpoint_counter = checkpoint.get('checkpoint_counter', 0)
        
        # Events are not saved (may contain unpicklable lambdas)
        # User should re-register events after loading
        self._events = {}
        
        # Rebuild graph from models
        self.graph = DependencyGraph()
        for process_id in self.models:
            self.graph.add_process(process_id)
        
        self._needs_reorder = True
    
    # =========================================================================
    # EXECUTION
    # =========================================================================
    
    def set_perturbation_manager(self, manager):
        self.perturbation_manager = manager
    
    def should_run(self, process_id, timescale):
        last_run = self.last_run_time.get(process_id, self.state.time - timescale.value)
        return (self.state.time - last_run) >= (timescale.value - 0.5)
    
    def step(self, global_dt=60.0):
        # Recompute execution order if needed (after add/remove)
        if self._needs_reorder:
            self.execution_order = self.graph.topological_sort()
            self._needs_reorder = False
        
        # 1. Apply perturbations first (can modify model params)
        if self.perturbation_manager:
            self.perturbation_manager.apply_all(self.state, global_dt)
        
        # 2. Execute processes in dependency order
        for process_id in self.execution_order:
            if process_id not in self.models:  # Skip if removed
                continue
            
            model = self.models[process_id]
            if self.should_run(process_id, model.timescale):
                model.step(self.state, model.timescale.value)
                self.last_run_time[process_id] = self.state.time
        
        # 3. Validate state (check constraints)
        self._validate_state()
        
        # 4. Check events (trigger actions if conditions met)
        self._check_events()
        
        # 5. Apply all pending deletions
        self._apply_deletions()
        
        # 6. Advance time
        self.state.time += global_dt
    
    def run(self, duration_seconds, global_dt=60.0, record_interval=300):
        self.execution_order = self.graph.topological_sort()
        target_time = self.state.time + duration_seconds
        last_record = 0
        
        print(f"Running simulation: {duration_seconds/3600:.1f}h")
        print()
        
        while self.state.time < target_time:
            self.step(global_dt)
            
            if self.state.time - last_record >= record_interval:
                self.state.record_history()
                last_record = self.state.time
            
            if int(self.state.time) % 3600 == 0:
                hours = int(self.state.time / 3600)
                glucose = self.state.get_signal('blood', 'glucose')
                insulin = self.state.get_signal('blood', 'insulin')
                pp = self.state.get_signal('blood', 'pancreatic_polypeptide')
                fed_status = self.state.get_organism_state('fed_status', 'unknown')
                print(f"  {hours}h: glucose={glucose:.1f} mg/dL, "
                      f"insulin={insulin:.1f} µU/mL, PP={pp:.1f} pg/mL, status={fed_status}")
        
        self.state.record_history()
        print()