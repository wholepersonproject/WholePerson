from core.graph import DependencyGraph

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
        
        # 3. Advance time
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