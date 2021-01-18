brainpy.integration package
===========================

.. currentmodule:: brainpy.integration
.. automodule:: brainpy.integration

.. contents::
    :local:
    :depth: 1



``diff_equation`` module
------------------------

.. autosummary::
    :toctree: _autosummary

    Expression
    DiffEquation


.. autoclass:: Expression
    :toctree:
    :members:

.. autoclass:: DiffEquation
   :members:


``integrator`` module
----------------------

.. autosummary::
    :toctree: _autosummary

    integrate
    get_integrator
    Integrator
    Euler
    Heun
    MidPoint
    RK2
    RK3
    RK4
    RK4Alternative
    ExponentialEuler
    MilsteinIto
    MilsteinStra

.. autoclass:: Integrator
   :members:

.. autoclass:: Euler
    :members:

.. autoclass:: Heun
    :members:

.. autoclass:: MidPoint
    :members:

.. autoclass:: RK2
    :members:

.. autoclass:: RK3
    :members:

.. autoclass:: RK4
    :members:

.. autoclass:: RK4Alternative
    :members:

.. autoclass:: ExponentialEuler
    :members:

.. autoclass:: MilsteinIto
    :members:

.. autoclass:: MilsteinStra
    :members:


``sympy_tools`` module
----------------------

.. autosummary::
    :toctree: _autosummary

    Parser
    Printer
    str2sympy
    sympy2str

.. autoclass:: Parser
    :members:

.. autoclass:: Printer
    :members:
