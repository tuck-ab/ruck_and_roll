import numpy as np
import pandas as pd

node_cols = ['node_num']
edge_cols = ['source', 'destination', 'weight']

class Graph:
    """Implementation of undirected weighted graph
    """
    def __init__(self, nodes, edges):
        self.node_df = pd.DataFrame(nodes, columns = node_cols)
        self.num_nodes = len(nodes)
        self.edge_df = pd.DataFrame(columns = edge_cols)
        
        '''for node in self._nodes:
            self._edges[node] = {}'''
        
        for n_from, n_to, weight in edges:
            self.add_edge(n_from, n_to, weight)
            
    def add_edge(self, n_from, n_to, weight):
        
        s1 = pd.DataFrame([[n_from, n_to, weight]], columns = edge_cols)
        s2 = pd.DataFrame([[n_to, n_from, weight]], columns = edge_cols)

        self.edge_df = self.edge_df.append(s1, ignore_index = True)
        self.edge_df = self.edge_df.append(s2, ignore_index = True)

        #pd.concat([self.edge_df, ], ignore_index=True)
        #pd.concat([self.edge_df, [n_to, n_from, weight]], ignore_index=True)
        
        #self._edges[n_from][n_to] = weight
        #self._edges[n_to][n_from] = weight
        
    def get_node_df(self):
        return self.node_df

    def get_edge_df(self):
        return self.edge_df

    def get_num_nodes(self):
        return self.num_nodes


    '''def get_edge_matrix(self):
        m = np.full((self._num_nodes, self._num_nodes), np.inf)
        
        for i, n1 in enumerate(self._nodes):
            for j, n2 in enumerate(self._nodes):
                if i == j:
                    m[i, j] = 0
                elif n2 in self._edges[n1]:
                    m[i, j] = self._edges[n1][n2]
                    
        return m, self._nodes'''
                    


