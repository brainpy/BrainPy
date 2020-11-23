# -*- coding: utf-8 -*-

from collections import namedtuple

import numpy as np
from numba import njit

__all__ = ['brentq']

_ECONVERGED = 0
_ECONVERR = -1

results = namedtuple('results', ['root', 'function_calls', 'iterations', 'converged'])


@njit
def _bisect_interval(a, b, fa, fb):
    """Conditional checks for intervals in methods involving bisection"""
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have different signs")
    root = 0.0
    status = _ECONVERR

    # Root found at either end of [a,b]
    if fa == 0:
        root = a
        status = _ECONVERGED
    if fb == 0:
        root = b
        status = _ECONVERGED

    return root, status


@njit
def brentq(f,
           a,
           b,
           args=(),
           xtol=2e-12,
           rtol=4 * np.finfo(float).eps,
           maxiter=100,
           disp=True):
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
    disp : bool, optional(default=True)
        If True, raise a RuntimeError if the algorithm didn't converge.

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

    fpre = f(xpre, *args)
    fcur = f(xcur, *args)
    funcalls = 2

    root, status = _bisect_interval(xpre, xcur, fpre, fcur)

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

                if (2 * abs(stry) < min(abs(spre), 3 * abs(sbis) - delta)):
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
            if (abs(scur) > delta):
                xcur += scur
            else:
                xcur += (delta if sbis > 0 else -delta)
            fcur = f(xcur, *args)
            funcalls += 1

    if disp and status == _ECONVERR:
        raise RuntimeError("Failed to converge")

    x, funcalls, iterations, flag = root, funcalls, itr, status
    return results(x, funcalls, iterations, flag == 0)
