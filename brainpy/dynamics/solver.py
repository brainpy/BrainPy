# -*- coding: utf-8 -*-

from collections import namedtuple

import numpy as np

__all__ = [
    'brentq',
    'find_root'
]

_ECONVERGED = 0
_ECONVERR = -1

results = namedtuple('results', ['root', 'function_calls', 'iterations', 'converged'])


def brentq(f, a, b, args=(), xtol=2e-12, maxiter=100,
           rtol=4 * np.finfo(float).eps):
    """
    Find a root of a function in a bracketing interval using Brent's method
    adapted from Scipy's brentq.

    Uses the classic Brent's method to find a zero of the function `f` on
    the sign changing interval [a , b].

    Parameters
    ----------
    f : callable
        Python function returning a number.  `f` must be continuous.
    a : number
        One end of the bracketing interval [a,b].
    b : number
        The other end of the bracketing interval [a,b].
    args : tuple, optional(default=())
        Extra arguments to be used in the function call.
    xtol : number, optional(default=2e-12)
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter must be nonnegative.
    rtol : number, optional(default=4*np.finfo(float).eps)
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root.
    maxiter : number, optional(default=100)
        Maximum number of iterations.

    Returns
    -------
    results : namedtuple

    """
    if xtol <= 0:
        raise ValueError("xtol is too small (<= 0)")
    if maxiter < 1:
        raise ValueError("maxiter must be greater than 0")

    # Convert to float
    xpre = a * 1.0
    xcur = b * 1.0

    # Conditional checks for intervals in methods involving bisection
    fpre = f(xpre, *args)
    fcur = f(xcur, *args)
    funcalls = 2

    if fpre * fcur > 0:
        raise ValueError("f(a) and f(b) must have different signs")
    root = 0.0
    status = _ECONVERR

    # Root found at either end of [a,b]
    if fpre == 0:
        root = xpre
        status = _ECONVERGED
    if fcur == 0:
        root = xcur
        status = _ECONVERGED

    root, status = root, status

    # Check for sign error and early termination
    if status == _ECONVERGED:
        itr = 0
    else:
        # Perform Brent's method
        for itr in range(maxiter):
            if fpre * fcur < 0:
                xblk = xpre
                fblk = fpre
                spre = scur = xcur - xpre
            if abs(fblk) < abs(fcur):
                xpre = xcur
                xcur = xblk
                xblk = xpre

                fpre = fcur
                fcur = fblk
                fblk = fpre

            delta = (xtol + rtol * abs(xcur)) / 2
            sbis = (xblk - xcur) / 2

            # Root found
            if fcur == 0 or abs(sbis) < delta:
                status = _ECONVERGED
                root = xcur
                itr += 1
                break

            if abs(spre) > delta and abs(fcur) < abs(fpre):
                if xpre == xblk:
                    # interpolate
                    stry = -fcur * (xcur - xpre) / (fcur - fpre)
                else:
                    # extrapolate
                    dpre = (fpre - fcur) / (xpre - xcur)
                    dblk = (fblk - fcur) / (xblk - xcur)
                    stry = -fcur * (fblk * dblk - fpre * dpre) / \
                           (dblk * dpre * (fblk - fpre))

                if 2 * abs(stry) < min(abs(spre), 3 * abs(sbis) - delta):
                    # good short step
                    spre = scur
                    scur = stry
                else:
                    # bisect
                    spre = sbis
                    scur = sbis
            else:
                # bisect
                spre = sbis
                scur = sbis

            xpre = xcur
            fpre = fcur
            if abs(scur) > delta:
                xcur += scur
            else:
                xcur += (delta if sbis > 0 else -delta)
            fcur = f(xcur, *args)
            funcalls += 1

    if status == _ECONVERR:
        raise RuntimeError("Failed to converge")

    x, funcalls, iterations = root, funcalls, itr

    return x


try:
    from numba import njit

    brentq = njit(brentq)
except ImportError:
    try:
        from scipy.optimize import brentq
    except ImportError:
        pass


def find_root(f, f_points, args=()):
    """Find the roots of the given function by numerical methods.

    Parameters
    ----------
    f : callable
        The function.
    f_points : onp.ndarray, list, tuple
        The value points.


    Returns
    -------
    roots : list
        The roots.
    """
    vals = f(f_points, *args)
    fs_len = len(f_points)
    fs_sign = np.sign(vals)

    roots = []
    fl_sign = fs_sign[0]
    f_i = 1
    while f_i < fs_len and fl_sign == 0.:
        roots.append(f_points[f_i - 1])
        fl_sign = fs_sign[f_i]
        f_i += 1
    while f_i < fs_len:
        fr_sign = fs_sign[f_i]
        if fr_sign == 0.:
            roots.append(f_points[f_i])
            if f_i + 1 < fs_len:
                fl_sign = fs_sign[f_i + 1]
            else:
                break
            f_i += 2
        else:
            if not np.isnan(fr_sign) and fl_sign != fr_sign:
                root = brentq(f, f_points[f_i - 1], f_points[f_i], args)
                roots.append(root)
            fl_sign = fr_sign
            f_i += 1

    return roots


if njit is not None:
    find_root = njit(find_root)
