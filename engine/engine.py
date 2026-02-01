import numpy as np
from core.graph import DependencyGraph
import pickle
import warnings

class PhysiologyEngine:
    """
    Multi-scale physiology simulation engine
    
    Features:
    - Dynamic process registration/removal during simulation
    - Automatic execution order computation based on dependencies
    - Input/output validation with auto-creation
    - Deletion queue system (safe, end-of-timestep deletion)
    - Constraints & validation (min/max bounds, clamping)
    - Checkpointing (save/load simulation state)
    - Multi-timescale adaptive stepping
    - Perturbation support
    """
    
    def __init__(self, state):
        self.state = state
        self.state._engine = self
        self.models = {}
        self.graph = DependencyGraph()
        self.execution_order = []
        self.last_run_time = {}
        self.perturbation_manager = None
        self._needs_reorder = True
        
        # Deletion queue system
        self._deletion_queue = []
        
        # Validation/constraints system
        self._constraints = {}  # (entity_id, signal_name) -> {min, max, clamp, warn}
        
        # Input/output validation
        self.inactive_processes = {}  # process_id -> reason
        
        # Checkpointing
        self._checkpoint_counter = 0
    
    # =========================================================================
    # PROCESS REGISTRATION
    # =========================================================================
    
    def register_model(self, process_id, model, dependencies=None):
        """
        Register a process (can be called during simulation)
        
        Args:
            process_id: Unique identifier
            model: ProcessModel instance
            dependencies: List of process_ids this depends on
        """
        self.models[process_id] = model
        self.graph.add_process(process_id)
        self.last_run_time[process_id] = -model.timescale.value
        
        if dependencies:
            for dep in dependencies:
                self.graph.add_dependency(dep, process_id)
        
        # Auto-create output signals if entity/organ exists
        self._ensure_outputs_exist(model)
        
        self._needs_reorder = True
        print(f"  ✓ Registered: {process_id}")
    
    def add_process(self, process_id, model, dependencies=None):
        """Alias for register_model (for consistency)"""
        self.register_model(process_id, model, dependencies)
    
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
            if process_id in self.inactive_processes:
                del self.inactive_processes[process_id]
            self._needs_reorder = True
            print(f"  ✗ Unregistered: {process_id}")
    
    def unregister_model(self, process_id):
        """Alias for remove_process (for consistency)"""
        self.remove_process(process_id)
    
    def has_process(self, process_id):
        """Check if a process exists"""
        return process_id in self.models
    
    def get_process(self, process_id):
        """Get a process by ID"""
        return self.models.get(process_id)
    
    def list_processes(self, active_only=False):
        """
        List all registered processes
        
        Args:
            active_only: If True, only show processes that can currently run
        
        Returns:
            list: Process IDs
        """
        if active_only:
            return [p for p in self.execution_order if p not in self.inactive_processes]
        return list(self.models.keys())
    
    def get_process_info(self, process_id):
        """
        Get detailed information about a process
        
        Args:
            process_id: Process to inspect
        
        Returns:
            dict: Process information including interface and status
        """
        if process_id not in self.models:
            return None
        
        model = self.models[process_id]
        can_run, missing_inputs = model.can_execute(self.state)
        can_write, missing_outputs = model.validate_outputs(self.state)
        
        return {
            'process_id': process_id,
            'interface': model.get_interface(),
            'can_execute': can_run,
            'missing_inputs': missing_inputs,
            'can_write': can_write,
            'missing_outputs': missing_outputs,
            'active': process_id not in self.inactive_processes,
            'last_run': self.last_run_time.get(process_id)
        }
    
    # =========================================================================
    # OUTPUT AUTO-CREATION
    # =========================================================================
    
    def _ensure_outputs_exist(self, model):
        """
        Ensure output signals exist in state (create with 0.0 if needed)
        
        This prevents silent failures when processes try to write to
        non-existent signals. Signals are auto-created if the target
        entity/organ/tissue exists.
        
        Uses smart lookup to find target in entities → organs → tissues
        
        Args:
            model: ProcessModel instance
        """
        if not hasattr(model, 'outputs') or not model.outputs:
            return
        
        for output_name, location in model.outputs.items():
            if len(location) == 2:
                # Smart lookup: ('blood', 'erythropoietin') or ('kidney', 'epo_capacity')
                target_id, signal_name = location
                
                # Try entities first
                if target_id in self.state.entities:
                    entity = self.state.entities[target_id]
                    
                    if entity['representation'] == 'lumped':
                        if signal_name not in entity['signals']:
                            entity['signals'][signal_name] = 0.0
                            print(f"  ℹ️  Created output signal: {target_id}.{signal_name}")
                    
                    elif entity['representation'] == 'spatial':
                        if signal_name not in entity['signals']:
                            shape = entity['shape']
                            entity['signals'][signal_name] = np.zeros(shape, dtype=np.float32)
                            print(f"  ℹ️  Created spatial output: {target_id}.{signal_name}")
                    continue
                
                # Try organs second
                if target_id in self.state.organs:
                    organ = self.state.organs[target_id]
                    
                    if 'signals' not in organ:
                        organ['signals'] = {}
                    
                    if signal_name not in organ['signals']:
                        # Check if spatial organ
                        if organ.get('representation') == 'spatial':
                            shape = organ['shape']
                            organ['signals'][signal_name] = np.zeros(shape, dtype=np.float32)
                            print(f"  ℹ️  Created spatial organ output: {target_id}.{signal_name}")
                        else:
                            organ['signals'][signal_name] = 0.0
                            print(f"  ℹ️  Created organ output: {target_id}.{signal_name}")
                    continue
                
                # Try tissues third
                if target_id in self.state.tissues:
                    tissue = self.state.tissues[target_id]
                    
                    if 'signals' not in tissue:
                        tissue['signals'] = {}
                    
                    if signal_name not in tissue['signals']:
                        tissue['signals'][signal_name] = 0.0
                        print(f"  ℹ️  Created tissue output: {target_id}.{signal_name}")
                    continue
                
                # Not found anywhere
                print(f"  ⚠️  Warning: Output target '{target_id}' not found for {model.process_id}")
            
            elif len(location) == 3 and location[2] == 'organism':
                # Organism state: ('age', None, 'organism')
                state_name = location[0]
                # Organism states are set directly, don't need pre-creation
                pass
    
    # =========================================================================
    # EXECUTION ORDER & VALIDATION
    # =========================================================================
    
    def _update_execution_order(self):
        """
        Recompute execution order and validate which processes can run
        
        This is called when:
        - Simulation starts
        - Processes are added/removed
        - State structure changes (organs/entities added/removed)
        """
        self.execution_order = self.graph.topological_sort()
        
        # Check which processes can execute based on inputs and outputs
        self.inactive_processes.clear()
        for process_id in self.execution_order:
            model = self.models[process_id]
            
            # Only validate if model has inputs/outputs declared
            if hasattr(model, 'can_execute'):
                can_run, missing_inputs = model.can_execute(self.state)
                
                if not can_run:
                    self.inactive_processes[process_id] = f"missing inputs: {missing_inputs}"
                    continue
            
            if hasattr(model, 'validate_outputs'):
                can_write, missing_outputs = model.validate_outputs(self.state)
                
                if not can_write:
                    self.inactive_processes[process_id] = f"missing output targets: {missing_outputs}"
        
        active_count = len(self.execution_order) - len(self.inactive_processes)
        print(f"  ⟳ Execution order: {active_count}/{len(self.execution_order)} active")
        
        if self.inactive_processes:
            print(f"  ⚠️  Inactive processes:")
            for proc_id, reason in self.inactive_processes.items():
                print(f"     {proc_id}: {reason}")
        
        self._needs_reorder = False
    
    def check_process_reactivation(self):
        """
        Check if any inactive processes can now run
        
        Called after each timestep to detect when:
        - Missing organs/entities have been added
        - Missing signals have been created
        
        Returns:
            list: Process IDs that were reactivated
        """
        newly_active = []
        
        for process_id in list(self.inactive_processes.keys()):
            model = self.models[process_id]
            
            can_run = True
            can_write = True
            
            if hasattr(model, 'can_execute'):
                can_run, _ = model.can_execute(self.state)
            
            if hasattr(model, 'validate_outputs'):
                can_write, _ = model.validate_outputs(self.state)
            
            if can_run and can_write:
                del self.inactive_processes[process_id]
                newly_active.append(process_id)
        
        if newly_active:
            print(f"  ✓ Reactivated: {newly_active}")
        
        return newly_active
    
    # =========================================================================
    # DELETION QUEUE SYSTEM
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
                    self._needs_reorder = True
            
            elif item_type == 'tissue':
                # args = (tissue_id,)
                tissue_id = args[0]
                if tissue_id in self.state.tissues:
                    del self.state.tissues[tissue_id]
                    self._needs_reorder = True
            
            elif item_type == 'organ':
                # args = (organ_id,)
                organ_id = args[0]
                if organ_id in self.state.organs:
                    del self.state.organs[organ_id]
                    self._needs_reorder = True
            
            elif item_type == 'process':
                # args = (process_id,)
                process_id = args[0]
                self.remove_process(process_id)
        
        # Clear queue
        self._deletion_queue.clear()
    
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
            warn: If True, print warnings on violations
        
        Example:
            # Glucose can't go negative
            engine.add_constraint('blood', 'glucose', min_val=0, clamp=True)
            
            # Warn if glucose too high (don't clamp)
            engine.add_constraint('blood', 'glucose', max_val=400, clamp=False, warn=True)
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
            
            if value is None:
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
        
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Reconnect engine
        self.state._engine = temp_engine
        
        print(f"✓ Saved checkpoint: {filename}")
        return filename
    
    def load_checkpoint(self, filename):
        """
        Load simulation state from checkpoint file
        
        Args:
            filename: Path to checkpoint file
        
        Example:
            engine.load_checkpoint('simulation_t1000.pkl')
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
        
        # Rebuild graph from models
        self.graph = DependencyGraph()
        for process_id in self.models:
            self.graph.add_process(process_id)
        
        self._needs_reorder = True
        
        print(f"✓ Loaded checkpoint: {filename}")
    
    # =========================================================================
    # PERTURBATION MANAGEMENT
    # =========================================================================
    
    def set_perturbation_manager(self, manager):
        """Set the perturbation manager"""
        self.perturbation_manager = manager
    
    # =========================================================================
    # EXECUTION
    # =========================================================================
    
    def should_run(self, process_id, timescale):
        """Check if a process should run based on its timescale"""
        last_run = self.last_run_time.get(process_id, self.state.time - timescale.value)
        return (self.state.time - last_run) >= (timescale.value - 0.5)
    
    def step(self, global_dt=60.0):
        """
        Execute one timestep
        
        Args:
            global_dt: Timestep in seconds
        """
        # Recompute execution order if needed (after add/remove or state changes)
        if self._needs_reorder:
            self._update_execution_order()
        
        # 1. Apply perturbations first (can modify model params)
        if self.perturbation_manager:
            self.perturbation_manager.apply_all(self.state, global_dt)
        
        # 2. Execute processes in dependency order
        for process_id in self.execution_order:
            # Skip if removed or inactive
            if process_id not in self.models:
                continue
            
            if process_id in self.inactive_processes:
                continue
            
            model = self.models[process_id]
            if self.should_run(process_id, model.timescale):
                model.step(self.state, model.timescale.value)
                self.last_run_time[process_id] = self.state.time
        
        # 3. Check if inactive processes can now run
        self.check_process_reactivation()
        
        # 4. Validate state (check constraints)
        self._validate_state()
        
        # 5. Apply all pending deletions
        self._apply_deletions()
        
        # 6. Advance time
        self.state.time += global_dt
    
    def run(self, duration_seconds, global_dt=60.0, record_interval=300, 
            perturbation_manager=None):
        """
        Run simulation
        
        Args:
            duration_seconds: How long to simulate (in seconds)
            global_dt: Base timestep (in seconds)
            record_interval: How often to save history snapshots (in seconds)
            perturbation_manager: Optional PerturbationManager
        
        Example:
            engine.run(duration_seconds=24*3600, global_dt=60.0, record_interval=300)
        """
        # Set perturbation manager if provided
        if perturbation_manager:
            self.perturbation_manager = perturbation_manager
        
        # Initial setup
        if self._needs_reorder:
            self._update_execution_order()
        
        target_time = self.state.time + duration_seconds
        last_record = self.state.time
        
        print(f"\nSimulation: {duration_seconds/3600:.1f}h @ dt={global_dt}s")
        print(f"Processes: {len(self.execution_order)}")
        if self.perturbation_manager:
            print(f"Perturbations: {len(self.perturbation_manager.perturbations)}")
        print("="*70)
        
        while self.state.time < target_time:
            self.step(global_dt)
            
            # Record history
            if self.state.time - last_record >= record_interval:
                self.state.record_history()
                last_record = self.state.time
        
        # Final snapshot
        self.state.record_history()
        
        elapsed = self.state.time - (target_time - duration_seconds)
        print(f"\n✓ Simulation complete: {elapsed/3600:.1f}h")
        print(f"  Final time: {self.state.time/3600:.1f}h")
        print(f"  History: {len(self.state.history)} snapshots")