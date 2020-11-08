import matplotlib.pyplot as plt
import numpy as np

import brainpy as bp


def show_weight(pre_ids, post_ids, weights, geometry, neu_id):
    height, width = geometry
    ids = np.where(pre_ids == neu_id)[0]
    post_ids = post_ids[ids]
    weights = weights[ids]

    X, Y = np.arange(height), np.arange(width)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros(geometry)
    for id_, weight in zip(post_ids, weights):
        h, w = id_ // width, id_ % width
        Z[h, w] = weight

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
    # surf = ax.plot_surface(X, Y, Z)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

import time

# t0 = time.time()
# gaussian_weight = bp.connect.GaussianWeight(
#     sigma=0.1, w_max=1., w_min=0.01,
#     normalize=True, include_self=True)
# pre_geom = post_geom = (100, 100)
# indices = np.arange(pre_geom[0] * pre_geom[1]).reshape(pre_geom)
# pre_ids, post_ids, weights = gaussian_weight(indices, indices)
# print(time.time() - t0)
# show_weight(pre_ids, post_ids, weights, pre_geom, 465)


t0 = time.time()
dog = bp.connect.DOG(
    sigmas=[0.08, 0.15],
    # sigmas=[0.01, 0.5],
    ws_max=[1.0, 0.7], w_min=0.01,
    normalize=True, include_self=True)
h = 100
pre_geom = post_geom = (h, h)
indices = np.arange(pre_geom[0] * pre_geom[1]).reshape(pre_geom)
pre_ids, post_ids, weights = dog(indices, indices)
print(time.time() - t0)
show_weight(pre_ids, post_ids, weights, (h, h), h * h // 2 + h // 2)

