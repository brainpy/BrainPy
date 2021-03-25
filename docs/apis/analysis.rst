brainpy.analysis
================

.. currentmodule:: brainpy.analysis
.. automodule:: brainpy.analysis

.. autosummary::
    :toctree: _autosummary

    BaseNeuronAnalyzer
    Base1DNeuronAnalyzer
    Base2DNeuronAnalyzer

    PhasePlane
    _PhasePlane1D
    _PhasePlane2D

    Bifurcation
    _Bifurcation1D
    _Bifurcation2D

    FastSlowBifurcation
    _FastSlow1D
    _FastSlow2D



.. autoclass:: BaseNeuronAnalyzer
    :members:

.. autoclass:: Base1DNeuronAnalyzer
    :members: get_f_dx, get_f_dfdx

.. autoclass:: Base2DNeuronAnalyzer
    :members: get_f_dy, get_f_dfdy, get_f_dgdx, get_f_dgdy, get_f_jacobian, get_y_by_x_in_y_eq, get_x_by_y_in_y_eq, get_y_by_x_in_x_eq, get_x_by_y_in_x_eq, get_f_fixed_point



.. autoclass:: PhasePlane
    :members: plot_fixed_point, plot_nullcline, plot_trajectory, plot_vector_field

.. autoclass:: _PhasePlane1D
    :members: plot_fixed_point, plot_nullcline, plot_trajectory, plot_vector_field

.. autoclass:: _PhasePlane2D
    :members: plot_fixed_point, plot_nullcline, plot_trajectory, plot_vector_field



.. autoclass:: Bifurcation
    :members: plot_bifurcation

.. autoclass:: _Bifurcation1D
    :members: plot_bifurcation

.. autoclass:: _Bifurcation2D
    :members: plot_bifurcation



.. autoclass:: FastSlowBifurcation
    :members: plot_bifurcation

.. autoclass:: _FastSlow1D
    :members: plot_bifurcation

.. autoclass:: _FastSlow2D
    :members: plot_bifurcation

