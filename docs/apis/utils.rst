npbrain.utils package
=====================

.. currentmodule:: npbrain.utils
.. automodule:: npbrain.utils

.. contents::
    :local:
    :depth: 2

``profile`` module
------------------

The setting of the overall framework.

Using the API in ``profile.py``, you can set

- the backend of numerical algorithm, ``numpy`` or ``numba``,
- the compilation options of JIT function in ``numba``,
- the precision of the numerical integration,
- the method of the numerical integration.


**Examples**

Set the default numerical integration precision.

.. code-block:: python

    from npbrain.utils import profile
    profile.set_dt(0.01)

Set the default numerical integration alorithm.

.. code-block:: python

         from npbrain.utils import profile
         profile.set_method('euler')

Or, you can use

.. code-block:: python

         from npbrain.core import forward_Euler
         profile.set_method(forward_Euler)

Set the default backend to ``numba`` and change the default JIT options.

.. code-block:: python

         from npbrain.utils import profile
         profile.set_backend('numba')
         profile.set_numba(nopython=True, fastmath=True, cache=True)


Methods for the settings of numerical integration.

.. autosummary::
    :toctree: _autosummary

    set_dt
    get_dt
    set_method
    set_method

Methods for the settings of computation backend.

.. autosummary::
    :toctree: _autosummary

    set_backend
    get_backend
    set_numba
    get_numba_profile



``connect`` module
------------------

The module helps you to construct the connectivity between neuron groups.
The coded several commonly used connection methods. Examples please see
the document of :doc:`Usage of connect module <usage_of_utils_connect>`.

The available methods of constructing synapse connection.

.. autosummary::
    :toctree: _autosummary

    one2one
    all2all
    grid_four
    grid_eight
    grid_N
    fixed_prob
    fixed_prenum
    fixed_postnum
    gaussian_weight
    gaussian_prob
    dog

The method of formatting connections.

.. autosummary::
    :toctree: _autosummary

    from_matrix
    from_ij


``helper`` module
-----------------

Helper of functions.

.. autosummary::
    :toctree: _autosummary

    jit_function
    jit_lambda
    autojit
    func_copy


Helper of computations.

.. autosummary::
    :toctree: _autosummary

    clip


``input_factory`` module
------------------------

The method of constructing current inputs.

Examples please see :doc:`Usage of input_factory module <usage_of_inputfactory>`.

.. autosummary::
    :toctree: _autosummary

    constant_current
    spike_current
    ramp_current


``run`` module
--------------

This module is useful for researchers. Because usually, we need grid-search
some parameters to known how the system perform in the parameters spaces.
This module takes advantage of multi-processing. It will save your much times.

Examples please see
`example 1 <https://github.com/chaoming0625/NumpyBrain/blob/master/npbrain/examples/networks/multi_process.py>`_
and
`example 2 <https://github.com/chaoming0625/NumpyBrain/blob/master/npbrain/examples/networks/multi_process_with_lock.py>`_.


.. autosummary::
    :toctree: _autosummary

    process_pool
    process_pool_lock


``visualize`` module
--------------------

This module provides commonly used functions to visualize model results.

.. autosummary::
    :toctree: _autosummary

    get_figure
    plot_value
    plot_potential
    plot_raster
    animation_potential

