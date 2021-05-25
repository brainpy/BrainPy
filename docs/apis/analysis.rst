``brainpy.analysis`` module
===========================

.. currentmodule:: brainpy.analysis
.. automodule:: brainpy.analysis

.. contents::
    :depth: 2

Summary
-------

.. autosummary::
    :toctree: _autosummary

    PhasePlane
    Bifurcation
    FastSlowBifurcation

    get_1d_stability_types
    get_2d_stability_types
    get_3d_stability_types
    stability_analysis


Phase Plane Analysis
--------------------

We provide a fundamental class `PhasePlane` to help users make
phase plane analysis for 1D/2D dynamical systems. Five methods
are provided, which can help you to plot:

- Fixed points
- Nullcline (zero-growth isoclines)
- Vector filed
- Limit cycles
- Trajectory


.. autoclass:: PhasePlane
    :members: plot_fixed_point, plot_nullcline, plot_trajectory, plot_vector_field, plot_limit_cycle_by_sim


Bifurcation Analysis
--------------------

We also provide basic bifurcation analysis for 1D/2D dynamical systems.


.. autoclass:: Bifurcation
    :members: plot_bifurcation, plot_limit_cycle_by_sim


Fast-slow System Analysis
-------------------------

For some 3D dynamical system, which can be treated as a fast-slow system,
they can be easily analyzed through our provided `FastSlowBifurcation`.

.. autoclass:: FastSlowBifurcation
    :members: plot_bifurcation


Useful Functions
----------------

In `brainpy.analysis` module, we also provide several useful functions
which may help your dynamical system analysis.

.. code-block:: python

    >>> get_1d_stability_types()
    ['saddle node', 'stable point', 'unstable point']

.. code-block:: python

    >>> get_2d_stability_types()
    ['saddle node',
     'center',
     'stable node',
     'stable focus',
     'stable star',
     'center manifold',
     'unstable node',
     'unstable focus',
     'unstable star',
     'unstable line',
     'stable degenerate',
     'unstable degenerate']

.. code-block:: python
    >>> get_3d_stability_types
    ['unclassified stable point',
     'unclassified unstable point',
     'stable node',
     'unstable saddle',
     'unstable node',
     'saddle node',
     'stable focus',
     'unstable focus',
     'unstable center',
     'unknown 3d']

