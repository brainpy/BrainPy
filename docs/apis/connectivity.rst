``brainpy.connect`` module
==========================

.. currentmodule:: brainpy.simulation.connectivity
.. automodule:: brainpy.simulation.connectivity


Formatter functions
-------------------

.. autosummary::
    :toctree: _autosummary

    ij2mat
    mat2ij
    pre2post
    post2pre
    pre2syn
    post2syn
    pre_slice
    post_slice


Regular Connections
-------------------

.. autosummary::
    :toctree: _autosummary

    One2One
    All2All
    GridFour
    GridEight
    GridN


.. autoclass:: One2One
   :members:

.. autoclass:: All2All
   :members:

.. autoclass:: GridFour
   :members:

.. autoclass:: GridEight
   :members:

.. autoclass:: GridN
   :members:


Random Connections
------------------

.. autosummary::
    :toctree: _autosummary

    FixedPostNum
    FixedPreNum
    FixedProb
    GaussianProb
    GaussianWeight
    DOG
    SmallWorld
    ScaleFreeBA
    ScaleFreeBADual
    PowerLaw



.. autoclass:: FixedPostNum
   :members:

.. autoclass:: FixedPreNum
   :members:

.. autoclass:: FixedProb
   :members:

.. autoclass:: GaussianProb
   :members:

.. autoclass:: GaussianWeight
   :members:

.. autoclass:: DOG
   :members:

.. autoclass:: SmallWorld
   :members:

.. autoclass:: ScaleFreeBA
   :members:

.. autoclass:: ScaleFreeBADual
   :members:

.. autoclass:: PowerLaw
   :members:
