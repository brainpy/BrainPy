# -*- coding: utf-8 -*-

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import numba
from brainpy import tools
from scipy.signal import convolve2d

@tools.numba_jit
def _get_nearest_dist(p2, n2, xs):
  if len(p2) == 1 and len(n2) == 1:
    return xs[p2[0]] - xs[n2[0]], xs[p2[0]], xs[n2[0]]  # dist,
  else:
    px = xs[p2]
    py = xs[n2].reshape((-1, 1))
    dists = px - py
    min_i = np.argmin(np.abs(dists))
    i, j = np.divmod(min_i, p2.shape[0])
    return dists[i, j], xs[i], xs[j]


@tools.numba_jit
def get_2d_box(fx_signs: np.ndarray, fy_signs: np.ndarray,
               xs: np.ndarray, ys: np.ndarray):
  signs = fx_signs[:-1] * fx_signs[1:] - fy_signs[:-1] * fy_signs[1:]
  x_len, y_len = fx_signs.shape
  boxes = []
  for i in range(y_len - 1):
    p2i = np.where(signs[:, i] == 2)[0]
    n2i = np.where(signs[:, i] == -2)[0]
    if len(p2i) == 0 or len(n2i) == 0:
      continue
    p2j = np.where(signs[:, i + 1] == 2)[0]
    n2j = np.where(signs[:, i + 1] == -2)[0]
    if len(p2j) == 0 or len(n2j) == 0:
      continue
    dist1, x1, x2 = _get_nearest_dist(p2i, n2i, xs)
    dist2, x3, x4 = _get_nearest_dist(p2j, n2j, xs)
    if dist1 * dist2 < 0:
      a = min(x1, x2, x3, x4)
      b = max(x1, x2, x3, x4)
      boxes.append(np.array([[a, b], [ys[i], ys[i + 1]]]))
  return boxes


gamma = 0.641  # Saturation factor for gating variable
tau = 0.06  # Synaptic time constant [sec]
tau0 = 0.002  # Noise time constant [sec]
a = 270.
b = 108.
d = 0.154

I0 = 0.3255  # background current [nA]
JE = 0.3725  # self-coupling strength [nA]
JI = -0.1137  # cross-coupling strength [nA]
JAext = 0.00117  # Stimulus input strength [nA]
sigma = 1.02  # nA

mu0 = 20.  # Stimulus firing rate [spikes/sec]
coh = 0.2  # # Stimulus coherence [%]
Ib1 = 0.3297
Ib2 = 0.3297


def int_s1(s1, s2):
  I1 = JE * s1 + JI * s2 + Ib1 + JAext * mu0 * (1. + coh)
  r1 = (a * I1 - b) / (1. - np.exp(-d * (a * I1 - b)))
  ds1dt = - s1 / tau + (1. - s1) * gamma * r1
  return ds1dt


def int_s2(s1, s2):
  I2 = JE * s2 + JI * s1 + Ib2 + JAext * mu0 * (1. - coh)
  r2 = (a * I2 - b) / (1. - np.exp(-d * (a * I2 - b)))
  ds2dt = - s2 / tau + (1. - s2) * gamma * r2
  return ds2dt


r = 0.001
s1s = np.arange(0., 1., r)
s2s = np.arange(0., 1., r)
Y, X = np.meshgrid(s2s, s1s)
s1_signs = np.sign(int_s1(X, Y))
s2_signs = np.sign(int_s2(X, Y))


# plt.imshow(s1_sings * s2_sings)
# plt.show()


def show1():
  # plt.pcolor(s1_signs[:-1] * s2_signs[1:], edgecolor='k')

  plt.figure()
  a = s1_signs * s2_signs
  plt.pcolor(a, edgecolor='k')
  plt.colorbar()

  # plt.figure()
  # b = convolve2d(a, np.ones((2, 2)))
  # plt.pcolor(b, )
  # plt.colorbar()
  plt.show()


def try1():
  aa = get_2d_box(s1_signs, s2_signs, s1s, s2s)
  for a in aa:
    print(a)

  bb = (s1_signs[:-1] * s1_signs[1:]) - (s2_signs[:-1] * s2_signs[1:])
  fig, ax = plt.subplots()
  # plt.imshow(bb.T[::-1])
  plt.imshow(bb.T)
  plt.colorbar()
  for a in aa:
    a = np.asarray(a * 1 / r, dtype=np.int_)
    rect = patches.Rectangle((a[0, 0], a[1, 0]),
                             a[0, 1] - a[0, 0],
                             a[1, 0] - a[1, 1],
                             linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
  plt.show()


try1()
