brainpy.dynamics package
========================

.. currentmodule:: brainpy.dynamics
.. automodule:: brainpy.dynamics

.. autosummary::
    :toctree: _autosummary

    NeuronDynamicsAnalyzer
    NeuronDynamics1D
    NeuronDynamics2D
    PhasePortraitAnalyzer
    BifurcationAnalyzer
    brentq
    find_root_of_1d
    find_root_of_2d
    stability_analysis
    rescale


.. autoclass:: NeuronDynamicsAnalyzer
    :members:


.. autoclass:: NeuronDynamics1D
    :members: get_f_dx, get_f_dfdx

.. autoclass:: NeuronDynamics2D
    :members: get_f_dy, get_f_dfdy, get_f_dgdx, get_f_dgdy, get_f_jacobian, get_y_by_x_in_y_eq, get_x_by_y_in_y_eq, get_y_by_x_in_x_eq, get_x_by_y_in_x_eq, get_f_fixed_point


.. autoclass:: PhasePortraitAnalyzer
    :members: plot_fixed_point, plot_nullcline, plot_trajectory, plot_vector_field


.. autoclass:: BifurcationAnalyzer
    :toctree:
    :members: plot_bifurcation

