brainpy.analysis package
========================

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

    brentq
    find_root_of_1d
    find_root_of_2d
    stability_analysis
    rescale



.. autoclass:: BaseNeuronAnalyzer
    :members:
    :toctree:

.. autoclass:: Base1DNeuronAnalyzer
    :members: get_f_dx, get_f_dfdx
    :toctree:

.. autoclass:: Base2DNeuronAnalyzer
    :members: get_f_dy, get_f_dfdy, get_f_dgdx, get_f_dgdy, get_f_jacobian, get_y_by_x_in_y_eq, get_x_by_y_in_y_eq, get_y_by_x_in_x_eq, get_x_by_y_in_x_eq, get_f_fixed_point
    :toctree:



.. autoclass:: PhasePlane
    :members: plot_fixed_point, plot_nullcline, plot_trajectory, plot_vector_field
    :toctree:

.. autoclass:: PhasePlane1D
    :members: plot_fixed_point, plot_nullcline, plot_trajectory, plot_vector_field
    :toctree:

.. autoclass:: PhasePlane2D
    :members: plot_fixed_point, plot_nullcline, plot_trajectory, plot_vector_field
    :toctree:



.. autoclass:: Bifurcation
    :members: plot_bifurcation
    :toctree:

.. autoclass:: _Bifurcation1D
    :members: plot_bifurcation
    :toctree:

.. autoclass:: _Bifurcation2D
    :members: plot_bifurcation
    :toctree:



.. autoclass:: FastSlowBifurcation
    :members: plot_bifurcation
    :toctree:

.. autoclass:: _FastSlow1D
    :members: plot_bifurcation
    :toctree:

.. autoclass:: _FastSlow2D
    :members: plot_bifurcation
    :toctree:

