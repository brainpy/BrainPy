brainpy.simulation
==================

.. currentmodule:: brainpy.simulation
.. automodule:: brainpy.simulation


Basic methods
-------------


.. autosummary::
    :toctree: _autosummary

    DynamicSystem
    ConstantDelay
    Monitor
    run_model


.. autoclass:: DynamicSystem
   :members: build, run, get_schedule, set_schedule


.. autoclass:: ConstantDelay
   :members:


.. autoclass:: Monitor
   :members:


Brain objects
-------------

.. autosummary::
    :toctree: _autosummary

    NeuGroup
    SynConn
    TwoEndConn
    Network

.. autoclass:: NeuGroup
   :members:

.. autoclass:: SynConn
   :members:

.. autoclass:: TwoEndConn
   :members:

.. autoclass:: Network
   :members: add, build, run


