import numpy as np
import math

from ..graph import Graph

class GraphGenerator:

    # Graph generation is per frame so take a bbstore which is one frame
    def __init__(self, bbstore):
        self.generate_graph(bbstore)

    def generate_graph(self, bbstore):
        """
        Generates a graph of the relevant bounding boxes in the frame

        Args:
            bbstore - An object of the BoundingBoxStore class
        Returns:
            An object of the graph class based on the bounding boxes provided
        """
        bbstore = bbstore.get_store()   # We don't require the extra functionality of the class from here

        # Get the bounding boxes related to people, filter out others
        # TODO: Should we also get the bbs for the ball? Maybe in a seperate graph?
        # TODO: Would need a way to differentiate from ball and person in graph
        filtered_bbs = []
        for bb in bbstore:
            if bb.get_class_and_score()[0] == "person":
                filtered_bbs.append(bb)

        # The upper bound for the number of nodes in the graph
        # The neural network takes a fixed size so our graph must be standardised in size
        # TODO: Examine YOLO on the games further to better determine this number
        max_bbs = 50

        nodes = np.zeros(max_bbs)
        edges = np.zeros((max_bbs, max_bbs))    # The graph is fully connected

        for i in range(0, min(max_bbs, len(filtered_bbs))):
            bb1 = filtered_bbs[i]
            midpoints = bb1.get_mid_point()
            node_name = str(midpoints[0]) + "," + str(midpoints[1])
            nodes[i] = node_name

            for j in range(0, min(max_bbs, len(filtered_bbs))):
                bb2 = filtered_bbs[j]
                dist = self.calculate_distance(bb1, bb2)
                edges[i][j] = dist

        # Need to provide node names for any nodes not reached in the loop
        for i in range(len(filtered_bbs), max_bbs):
            nodes[i] = -i   # No other nodes will be prefixed by a '-' so there are no conflicts

        self.graph = Graph(nodes, edges)
        return self.graph

    def calculate_distance(self, bb1, bb2):
        pass

    def get_graph(self):
        """
        Getter method for the graph generator

        Returns:
            The graph object stored in this generator
        """
        return self.graph
