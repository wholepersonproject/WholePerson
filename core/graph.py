class DependencyGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
    
    def add_process(self, process_id):
        self.nodes[process_id] = {}
        self.edges[process_id] = []
    
    def add_dependency(self, source, target):
        if source not in self.edges:
            self.edges[source] = []
        self.edges[source].append(target)
    
    def find_strongly_connected_components(self):
        """
        Find strongly connected components using Tarjan's algorithm
        Returns: List of SCCs, where each SCC is a list of node IDs
        """
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        on_stack = {}
        sccs = []
        
        def strongconnect(node):
            index[node] = index_counter[0]
            lowlinks[node] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)
            on_stack[node] = True
            
            # Consider successors
            for successor in self.edges.get(node, []):
                if successor not in index:
                    strongconnect(successor)
                    lowlinks[node] = min(lowlinks[node], lowlinks[successor])
                elif on_stack.get(successor, False):
                    lowlinks[node] = min(lowlinks[node], index[successor])
            
            # If node is a root node, pop the stack and generate an SCC
            if lowlinks[node] == index[node]:
                scc = []
                while True:
                    successor = stack.pop()
                    on_stack[successor] = False
                    scc.append(successor)
                    if successor == node:
                        break
                sccs.append(scc)
        
        for node in self.nodes:
            if node not in index:
                strongconnect(node)
        
        return sccs
    
    def topological_sort(self):
        """
        Topological sort with proper SCC handling:
        1. Find strongly connected components (SCCs)
        2. Do topological sort on the condensed graph of SCCs
        3. Within each SCC, use arbitrary order (cycles resolve via time)
        """
        # Find SCCs
        sccs = self.find_strongly_connected_components()
        
        # Check if any SCC has more than one node (indicates cycle)
        cycles_detected = any(len(scc) > 1 for scc in sccs)
        
        if cycles_detected:
            print(f"WARNING: Circular dependencies detected!")
            for scc in sccs:
                if len(scc) > 1:
                    print(f"         Cycle: {sorted(scc)}")
            print(f"         Running cyclic processes in arbitrary order")
            print(f"         (cycles resolve via time delays)")
        
        # Build condensed graph where each SCC is a single node
        scc_map = {}  # node -> scc_index
        for scc_idx, scc in enumerate(sccs):
            for node in scc:
                scc_map[node] = scc_idx
        
        # Build edges between SCCs
        scc_edges = {i: set() for i in range(len(sccs))}
        for source in self.edges:
            source_scc = scc_map[source]
            for target in self.edges[source]:
                target_scc = scc_map[target]
                if source_scc != target_scc:  # Only edges between different SCCs
                    scc_edges[source_scc].add(target_scc)
        
        # Topological sort on SCCs
        scc_in_degree = {i: 0 for i in range(len(sccs))}
        for source_scc in scc_edges:
            for target_scc in scc_edges[source_scc]:
                scc_in_degree[target_scc] += 1
        
        scc_queue = [i for i in range(len(sccs)) if scc_in_degree[i] == 0]
        scc_order = []
        
        while scc_queue:
            scc_idx = scc_queue.pop(0)
            scc_order.append(scc_idx)
            
            for target_scc in scc_edges[scc_idx]:
                scc_in_degree[target_scc] -= 1
                if scc_in_degree[target_scc] == 0:
                    scc_queue.append(target_scc)
        
        # Build final execution order
        result = []
        for scc_idx in scc_order:
            scc = sccs[scc_idx]
            # Sort nodes within SCC alphabetically for determinism
            result.extend(sorted(scc))
        
        return result