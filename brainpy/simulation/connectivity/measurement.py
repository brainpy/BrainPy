# -*- coding: utf-8 -*-


import numpy as np

__all__ = [
    'cluster_coefficient',
]


def cluster_coefficient(conn_mat):
    ccs = []
    nodes = np.arange(conn_mat.shape[0])
    for node in nodes:
        triangles = 0
        neighbor_pairs = 0
        neighbors = np.where(conn_mat[node])[0]
        for neighbor1 in neighbors:
            for neighbor2 in neighbors:
                if neighbor1 == neighbor2:
                    continue
                neighbor_pairs += 1
                if conn_mat[neighbor1, neighbor2]:
                    triangles += 1
        ccs.append(triangles / neighbor_pairs)
    return sum(ccs) / len(ccs)


def average_path_length(conn_mat):
    def _get_sample_paths():
        node_count = len(self._random_graph)
        sample_paths = []
        if node_count <= 14:  # with 14 nodes, there are a maximum of 91 node combinations possible, 15 has 105
            for node1 in list(self._random_graph.nodes):
                for node2 in list(self._random_graph.nodes):
                    if node1 != node2 and (node1, node2) not in sample_paths and (node2, node1) not in sample_paths:
                        sample_paths.append((node1, node2))
        else:
            while len(sample_paths) < 100:
                random_node1 = list(self._random_graph.nodes)[random.randint(0, node_count - 1)]
                random_node2 = list(self._random_graph.nodes)[random.randint(0, node_count - 1)]
                if (random_node1 != random_node2 and (random_node1, random_node2) not in sample_paths and
                        (random_node2, random_node1) not in sample_paths):
                    sample_paths.append((random_node1, random_node2))
        return sample_paths

    def _compute_average_path_length(self):
        sample_paths = self._get_sample_paths()
        distances = []
        for (first_node, second_node) in sample_paths:
            distances.append(nx.shortest_path_length(self._random_graph, first_node, second_node))
        self._average_path_length = sum(distances) / len(distances)
