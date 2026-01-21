from core.graph import DependencyGraph

class PhysiologyEngine:
    def __init__(self, state):
        self.state = state
        self.state._engine = self  # Allow perturbations to access models
        self.models = {}
        self.graph = DependencyGraph()
        self.execution_order = []
        self.last_run_time = {}
        self.perturbation_manager = None
    
    def register_model(self, process_id, model, dependencies=None):
        self.models[process_id] = model
        self.graph.add_process(process_id)
        self.last_run_time[process_id] = -model.timescale.value
        
        if dependencies:
            for dep in dependencies:
                self.graph.add_dependency(dep, process_id)
    
    def set_perturbation_manager(self, manager):
        self.perturbation_manager = manager
    
    def should_run(self, process_id, timescale):
        last_run = self.last_run_time.get(process_id, self.state.time - timescale.value)
        return (self.state.time - last_run) >= (timescale.value - 0.5)
    
    def step(self, global_dt=60.0):
        # 1. Apply perturbations first (can modify model params)
        if self.perturbation_manager:
            self.perturbation_manager.apply_all(self.state, global_dt)
        
        # 2. Execute processes in dependency order
        for process_id in self.execution_order:
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
                fed_status = self.state.get_organism_state('fed_status', 'unknown')
                print(f"  {hours}h: glucose={glucose:.1f} mg/dL, "
                      f"insulin={insulin:.1f} µU/mL, status={fed_status}")
        
        self.state.record_history()
        print()