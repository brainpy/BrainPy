Dynamics Analysis (Symbolic)
============================

.. currentmodule:: brainpy.analysis.symbolic
.. automodule:: brainpy.analysis.symbolic


.. autosummary::
    :toctree: generated/

    PhasePlane
    Bifurcation
    FastSlowBifurcation



We provide a fundamental class ``PhasePlane`` to help users make
phase plane analysis for 1D/2D dynamical systems. Five methods
are provided, which can help you to plot:

- Fixed points
- Nullcline (zero-growth isoclines)
- Vector filed
- Limit cycles
- Trajectory


.. autoclass:: PhasePlane
    :members: plot_fixed_point, plot_nullcline, plot_trajectory, plot_vector_field, plot_limit_cycle_by_sim



We also provide basic bifurcation analysis for 1D/2D dynamical systems.


.. autoclass:: Bifurcation
    :members: plot_bifurcation, plot_limit_cycle_by_sim



For some 3D dynamical system, which can be treated as a fast-slow system,
they can be easily analyzed through our provided ``FastSlowBifurcation``.

.. autoclass:: FastSlowBifurcation
    :members: plot_bifurcation


