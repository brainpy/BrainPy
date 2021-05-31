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

