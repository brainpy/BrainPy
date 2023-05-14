# -*- coding: utf-8 -*-

import numpy as np

__all__ = [
  'get_1d_stability_types',
  'get_2d_stability_types',
  'get_3d_stability_types',


  'stability_analysis',

  'CENTER_MANIFOLD',
  'SADDLE_NODE',
  'STABLE_POINT_1D',
  'UNSTABLE_POINT_1D',

  'CENTER_2D',
  'STABLE_NODE_2D',
  'STABLE_FOCUS_2D',
  'STABLE_STAR_2D',
  'STABLE_DEGENERATE_2D',
  'UNSTABLE_NODE_2D',
  'UNSTABLE_FOCUS_2D',
  'UNSTABLE_STAR_2D',
  'UNSTABLE_DEGENERATE_2D',
  'UNSTABLE_LINE_2D',
]



SADDLE_NODE = 'saddle node'
CENTER_MANIFOLD = 'center manifold'

STABLE_POINT_1D = 'stable point'
UNSTABLE_POINT_1D = 'unstable point'

CENTER_2D = 'center'
STABLE_NODE_2D = 'stable node'
STABLE_FOCUS_2D = 'stable focus'
STABLE_STAR_2D = 'stable star'
STABLE_DEGENERATE_2D = 'stable degenerate'
UNSTABLE_NODE_2D = 'unstable node'
UNSTABLE_FOCUS_2D = 'unstable focus'
UNSTABLE_STAR_2D = 'unstable star'
UNSTABLE_DEGENERATE_2D = 'unstable degenerate'
UNSTABLE_LINE_2D = 'unstable line'


STABLE_POINT_3D = 'unclassified stable point'
UNSTABLE_POINT_3D = 'unclassified unstable point'
STABLE_NODE_3D = 'stable node'
UNSTABLE_SADDLE_3D = 'unstable saddle'
UNSTABLE_NODE_3D = 'unstable node'
STABLE_FOCUS_3D = 'stable focus'
UNSTABLE_FOCUS_3D = 'unstable focus'
UNSTABLE_CENTER_3D = 'unstable center'
UNKNOWN_3D = 'unknown 3d'


def get_1d_stability_types():
  """Get the stability types of 1D system."""
  return [SADDLE_NODE, STABLE_POINT_1D, UNSTABLE_POINT_1D]


def get_2d_stability_types():
  """Get the stability types of 2D system."""
  return [SADDLE_NODE, CENTER_2D, STABLE_NODE_2D, STABLE_FOCUS_2D,
          STABLE_STAR_2D, CENTER_MANIFOLD, UNSTABLE_NODE_2D,
          UNSTABLE_FOCUS_2D, UNSTABLE_STAR_2D, UNSTABLE_LINE_2D,
          STABLE_DEGENERATE_2D, UNSTABLE_DEGENERATE_2D]


def get_3d_stability_types():
  """Get the stability types of 3D system."""
  return [STABLE_POINT_3D, UNSTABLE_POINT_3D, STABLE_NODE_3D,
          UNSTABLE_SADDLE_3D, UNSTABLE_NODE_3D, SADDLE_NODE,
          STABLE_FOCUS_3D, UNSTABLE_FOCUS_3D, UNSTABLE_CENTER_3D, UNKNOWN_3D]


def stability_analysis(derivatives):
  """Stability analysis of fixed points for low-dimensional system.

  The analysis is referred to [1]_.

  Parameters
  ----------
  derivatives : float, tuple, list, np.ndarray
      The derivative of the f.

  Returns
  -------
  fp_type : str
      The type of the fixed point.

  References
  ----------

  .. [1] http://www.egwald.ca/nonlineardynamics/twodimensionaldynamics.php

  """
  if np.size(derivatives) == 1:  # 1D dynamical system
    if derivatives == 0:
      return SADDLE_NODE
    elif derivatives > 0:
      return UNSTABLE_POINT_1D
    else:
      return STABLE_POINT_1D

  elif np.size(derivatives) == 4:  # 2D dynamical system
    a = derivatives[0][0]
    b = derivatives[0][1]
    c = derivatives[1][0]
    d = derivatives[1][1]

    # trace
    p = a + d
    # det
    q = a * d - b * c

    # judgement
    if q < 0:
      return SADDLE_NODE
    elif q == 0:
      if p <= 0:
        return CENTER_MANIFOLD
      else:
        return UNSTABLE_LINE_2D
    else:
      # parabola
      e = p * p - 4 * q
      if p == 0:
        return CENTER_2D
      elif p > 0:
        if e < 0:
          return UNSTABLE_FOCUS_2D
        elif e > 0:
          return UNSTABLE_NODE_2D
        else:
          w = np.linalg.eigvals(derivatives)
          if w[0] == w[1]:
            return UNSTABLE_DEGENERATE_2D
          else:
            return UNSTABLE_STAR_2D
      else:
        if e < 0:
          return STABLE_FOCUS_2D
        elif e > 0:
          return STABLE_NODE_2D
        else:
          w = np.linalg.eigvals(derivatives)
          if w[0] == w[1]:
            return STABLE_DEGENERATE_2D
          else:
            return STABLE_STAR_2D

  elif np.size(derivatives) == 9:  # 3D dynamical system
    eigenvalues = np.linalg.eigvals(np.array(derivatives))
    is_real = np.isreal(eigenvalues)
    if is_real.all():
      eigenvalues = np.sort(eigenvalues)
      if eigenvalues[2] < 0:
        return STABLE_NODE_3D
      elif eigenvalues[2] == 0:
        return UNKNOWN_3D
      else:
        if eigenvalues[0] > 0:
          return UNSTABLE_NODE_3D
        elif eigenvalues[0] == 0:
          return UNKNOWN_3D
        else:
          if eigenvalues[1] < 0:
            return SADDLE_NODE
          elif eigenvalues[1] == 0:
            return UNKNOWN_3D
          else:
            return UNSTABLE_SADDLE_3D
    else:
      if is_real.sum() == 1:
        real_id = np.where(is_real)[0]
        non_real_id = np.where(np.logical_not(is_real))[0]
        v0 = eigenvalues[real_id]
        v1 = eigenvalues[non_real_id[0]]
        v2 = eigenvalues[non_real_id[1]]
        v1_real = np.real(v1)
        assert np.conj(v1) == v2
        if v0 < 0:
          if v1_real < 0:
            return STABLE_FOCUS_3D
          elif v1_real == 0:  # 零实部
            return UNKNOWN_3D
          else:
            return UNSTABLE_FOCUS_3D
        elif v0 == 0:
          if v1_real <= 0:
            return UNKNOWN_3D  # 零实部
          else:
            return UNSTABLE_POINT_3D  # TODO
        else:
          if v1_real < 0:
            return UNSTABLE_FOCUS_3D
          elif v1_real == 0:
            return UNSTABLE_CENTER_3D
          else:
            return UNSTABLE_POINT_3D  # TODO
      # else:
      #   raise ValueError()

    eigenvalues = np.real(eigenvalues)
    if np.all(eigenvalues < 0):
      return STABLE_POINT_3D  # TODO
    else:
      return UNSTABLE_POINT_3D  # TODO
  else:
    raise ValueError('Unknown derivatives, only supports the jacobian '
                     'matrix with the shape of (1), (2, 2), or (3, 3).')
