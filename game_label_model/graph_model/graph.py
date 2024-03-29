import numpy as np

class Graph:
    """Implementation of undirected weighted graph
    """
    def __init__(self, nodes, edges):
        self._nodes = nodes
        self._num_nodes = len(nodes)
        self._edges = {}
        
        for node in self._nodes:
            self._edges[node] = {}
        
        for n_from, n_to, weight in edges:
            self.add_edge(n_from, n_to, weight)

    def __init__(self, nodes, edges, overload):
        self._nodes = nodes
        self._num_nodes = len(nodes)
        self._edges = {}
        
        for node in self._nodes:
            self._edges[node] = {}

        for i in range(0, len(edges)):
            name_i = nodes[i]
            for j in range(0, len(edges)):
                name_j = nodes[j]
                self.add_edge(name_i, name_j, edges[i][j])

            
    def add_edge(self, n_from, n_to, weight):
        self._edges[n_from][n_to] = weight
        self._edges[n_to][n_from] = weight
        
    def get_edge_matrix(self):
        m = np.full((self._num_nodes, self._num_nodes), np.inf)
        
        for i, n1 in enumerate(self._nodes):
            for j, n2 in enumerate(self._nodes):
                if n2 in self._edges[n1]:
                    m[i, j] = self._edges[n1][n2]
                    
        return m, self._nodes