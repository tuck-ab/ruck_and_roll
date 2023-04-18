import numpy as np
import math

from .graph import Graph

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
        filtered_bbs = []
        has_ball = False
        for bb in bbstore:
            if bb.get_class_and_score()[0] == "person":
                filtered_bbs.append(bb)
            elif bb.get_class_and_score()[0] == "ball":
                has_ball = True
                ball = bb
        if has_ball:
            filtered_bbs.append(ball)

        # The upper bound for the number of nodes in the graph
        # The neural network takes a fixed size so our graph must be standardised in size
        max_bbs = 50

        nodes = np.empty(max_bbs, dtype=object)
        edges = np.zeros((max_bbs, max_bbs))    # The graph is fully connected
        for i in range(0, max_bbs):
            for j in range(0, max_bbs):
                edges[i][j] = -1

        for i in range(0, min(max_bbs, len(filtered_bbs))):
            bb1 = filtered_bbs[i]
            midpoints = bb1.get_mid_point()
            node_name = str(midpoints[0]) + "," + str(midpoints[1])
            i_val = i
            if i == len(filtered_bbs) - 1 and has_ball:
                nodes[max_bbs - 1] = node_name
                i_val = max_bbs - 1
            else:
                nodes[i] = node_name

            for j in range(0, min(max_bbs, len(filtered_bbs))):
                bb2 = filtered_bbs[j]
                dist = self.calculate_distance(bb1, bb2)
                j_val = j
                if j == len(filtered_bbs) - 1 and has_ball:
                    j_val = max_bbs - 1
                edges[i_val][j_val] = dist

        # Need to provide node names for any nodes not reached in the loop
        start = len(filtered_bbs) - 1 if has_ball else len(filtered_bbs)
        stop = max_bbs - 1 if has_ball else max_bbs
        for i in range(start, stop):
            nodes[i] = str(i * -1)   # No other nodes will be prefixed by a '-' so there are no conflicts

        self.graph = Graph(nodes, edges, True)
        return self.graph

    def calculate_distance(self, bb1, bb2):
        """
        Calcualtes the distance of the two bounding boxes for the edge weight in the graph
        Args:
            bb1 - The first bounding box
            bb2 - The second bounding box
        Returns:
            A float representing the physical distance between bb1 and bb2
        """
        w1, h1 = bb1.get_width_height()
        _, h2 = bb2.get_width_height()

        # Background people will appear smaller than foreground people through parralax
        # Therefore their distance must be greater
        # This means that a form of 3D data must be extracted from the 2D image
        # This 'z-distance' will be calculated using the relative heights of the players. This is less likely to be obscured than their width

        ratio = h1 / h2 if h1 > h2 else h2 / h1 # Ratio is always >= 1 to be multiplicative of the distance. Smaller distance is closer

        mp1 = bb1.get_mid_point()
        mp2 = bb2.get_mid_point()

        # Only in the case where the boxes are the same will the second ratio be returned
        # This allows us to take account of how horizontal a player is
        val = math.sqrt((mp1[0] - mp2[0]) ** 2.0 + (mp1[1] - mp2[1]) ** 2.0)
        dist = max(val * ratio, w1 / h1)
        return dist

    def get_graph(self):
        """
        Getter method for the graph generator
        Returns:
            The graph object stored in this generator
        """
        return self.graph