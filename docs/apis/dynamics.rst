brainpy.dynamics package
========================

.. currentmodule:: brainpy.dynamics
.. automodule:: brainpy.dynamics

.. autosummary::
    :toctree: _autosummary

    BaseNeuronAnalyzer
    Base1DNeuronAnalyzer
    Base2DNeuronAnalyzer

    PhasePlaneAnalyzer
    PhasePortraitAnalyzer
    PhasePlane1DAnalyzer
    PhasePlane2DAnalyzer

    BifurcationAnalyzer
    Bifurcation1DAnalyzer
    Bifurcation2DAnalyzer

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



.. autoclass:: PhasePlaneAnalyzer
    :members: plot_fixed_point, plot_nullcline, plot_trajectory, plot_vector_field
    :toctree:

.. autoclass:: PhasePortraitAnalyzer

.. autoclass:: PhasePlane1DAnalyzer
    :members: plot_fixed_point, plot_nullcline, plot_trajectory, plot_vector_field
    :toctree:

.. autoclass:: PhasePlane2DAnalyzer
    :members: plot_fixed_point, plot_nullcline, plot_trajectory, plot_vector_field
    :toctree:



.. autoclass:: BifurcationAnalyzer
    :members: plot_bifurcation
    :toctree:

.. autoclass:: Bifurcation1DAnalyzer
    :members: plot_bifurcation
    :toctree:

.. autoclass:: Bifurcation2DAnalyzer
    :members: plot_bifurcation
    :toctree:

