# -*- coding: utf-8 -*-

from brainpy import backend
from brainpy.integrators import constants
from .wrapper import exp_euler_wrapper

__all__ = [
    'exponential_euler',
]


def exponential_euler(f=None, show_code=None, dt=None, var_type=None, im_return=()):
    """First order, explicit exponential Euler method.

    For an ODE equation of the form

    .. math::

        y^{\\prime}=f(y), \quad y(0)=y_{0}

    its schema is given by

    .. math::

        y_{n+1}= y_{n}+h \\varphi(hA) f (y_{n})

    where :math:`A=f^{\prime}(y_{n})` and :math:`\\varphi(z)=\\frac{e^{z}-1}{z}`.

    For linear ODE system: :math:`y^{\\prime} = Ay + B`,
    the above equation is equal to

    .. math::

        y_{n+1}= y_{n}e^{hA}-B/A(1-e^{hA})

    Parameters
    ----------

    Returns
    -------
    func : callable
        The one-step numerical integrator function.
    """

    dt = backend.get_dt() if dt is None else dt
    show_code = False if show_code is None else show_code
    var_type = constants.SCALAR_VAR if var_type is None else var_type

    if f is None:
        return lambda f: exp_euler_wrapper(f, show_code=show_code, dt=dt,
                                           var_type=var_type, im_return=im_return)
    else:
        return exp_euler_wrapper(f, show_code=show_code, dt=dt,
                                 var_type=var_type, im_return=im_return)
