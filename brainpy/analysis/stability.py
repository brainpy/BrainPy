# -*- coding: utf-8 -*-

import numpy as np


CENTER_MANIFOLD = 'center manifold'
SADDLE_NODE = 'saddle node'
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


def get_1d_classification():
    return [SADDLE_NODE, STABLE_POINT_1D, UNSTABLE_POINT_1D]


def get_2d_classification():
    return [SADDLE_NODE, CENTER_2D, STABLE_NODE_2D, STABLE_FOCUS_2D,
            STABLE_STAR_2D, CENTER_MANIFOLD, UNSTABLE_NODE_2D,
            UNSTABLE_FOCUS_2D, UNSTABLE_STAR_2D, UNSTABLE_LINE_2D,
            STABLE_DEGENERATE_2D, UNSTABLE_DEGENERATE_2D]


def get_3d_classification():
    return []


def stability_analysis(derivative):
    """Stability analysis for fixed point [1]_.

    Parameters
    ----------
    derivative : float, tuple, list, np.ndarray
        The derivative of the f.

    Returns
    -------
    fp_type : str
        The type of the fixed point.

    References
    ----------

    .. [1] http://www.egwald.ca/nonlineardynamics/twodimensionaldynamics.php

    """
    if np.size(derivative) == 1:
        if derivative == 0:
            return SADDLE_NODE
        elif derivative > 0:
            return STABLE_POINT_1D
        else:
            return UNSTABLE_POINT_1D

    elif np.size(derivative) == 4:
        a = derivative[0][0]
        b = derivative[0][1]
        c = derivative[1][0]
        d = derivative[1][1]

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
                    w = np.linalg.eigvals(derivative)
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
                    w = np.linalg.eigvals(derivative)
                    if w[0] == w[1]:
                        return STABLE_DEGENERATE_2D
                    else:
                        return STABLE_STAR_2D

    elif np.size(derivative) == 9:
        pass

    else:
        raise ValueError('Unknown derivatives, only supports the jacobian '
                         'matrix with the shape of(1), (2, 2), or (3, 3).')
