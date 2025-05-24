import time

class StateGraph:
    """Graph to store environment states and transitions"""
    def __init__(self):
        self.nodes = []
        self.current_node = None
        self.edges = {}  # Map from node index to list of connected node indices
        self.node_metadata = {}  # Additional data associated with each node

    def add_node(self, node):
        """Add a node to the graph and set it as current"""
        self.nodes.append(node)
        node_idx = len(self.nodes) - 1
        self.current_node = node
        
        # Initialize edges for new node
        if node_idx not in self.edges:
            self.edges[node_idx] = []
        
        # Connect to previous node if exists
        if node_idx > 0:
            self.add_edge(node_idx - 1, node_idx)
            
        return node_idx
        
    def add_edge(self, from_idx, to_idx):
        """Add a directed edge between nodes"""
        if from_idx in self.edges:
            self.edges[from_idx].append(to_idx)
            
    def get_node(self, idx):
        """Get node by index"""
        if 0 <= idx < len(self.nodes):
            return self.nodes[idx]
        return None
        
    def find_path(self, start_idx, end_idx):
        """Find a path between two nodes using BFS"""
        visited = set()
        queue = [[start_idx]]
        
        if start_idx == end_idx:
            return [start_idx]
            
        while queue:
            path = queue.pop(0)
            node = path[-1]
            
            if node not in visited:
                for neighbor in self.edges.get(node, []):
                    new_path = list(path)
                    new_path.append(neighbor)
                    
                    if neighbor == end_idx:
                        return new_path
                        
                    queue.append(new_path)
                    
                visited.add(node)
                
        return None  # No path found
        
    def add_metadata(self, node_idx, key, value):
        """Associate metadata with a node"""
        if node_idx not in self.node_metadata:
            self.node_metadata[node_idx] = {}
        self.node_metadata[node_idx][key] = value
        
    def get_metadata(self, node_idx, key=None):
        """Get metadata for a node"""
        if node_idx not in self.node_metadata:
            return None
            
        if key is None:
            return self.node_metadata[node_idx]
        return self.node_metadata[node_idx].get(key)
        
    def current_node_idx(self):
        """Get index of current node"""
        if self.current_node is None:
            return None
        return self.nodes.index(self.current_node)


class EnvironmentState:
    """Container for environment state data"""
    def __init__(self):
        self.vehicle_state = {}
        self.env_state = {}
        self.timestamp = time.time()