``brainpy.version2.neurons`` module
===================================

.. currentmodule:: brainpy.version2.neurons
.. automodule:: brainpy.version2.neurons

.. contents::
   :local:
   :depth: 1


From ``brainpy>=2.4.3``, most of models in ``brainpy.version2.neurons`` have been
reimplemented with ``brainpy.version2.dyn`` module.

However, ``brainpy.version2.neurons`` is still independent from ``brainpy.version2.dyn`` module.

The most significant difference between models in ``brainpy.version2.neurons`` and ``brainpy.version2.dyn`` is that:

- the former only support the integration style without liquid time constant (which means that
  the time constants in these neuron models are fixed once initialization)
- the former supports the integration with SDE by specifying the ``noise`` parameter. For example,
  ``brainpy.version2.neurons.HH(size, ..., noise=1.)``
- the former has one additional ``input`` variable for receiving external inputs.


Biological Models
-----------------

.. autosummary::
   :toctree: generated/

   HH
   MorrisLecar
   PinskyRinzelModel
   WangBuzsakiModel


Fractional-order Models
-----------------------

.. autosummary::
   :toctree: generated/

   FractionalNeuron
   FractionalFHR
   FractionalIzhikevich


Reduced Models
--------------

.. autosummary::
   :toctree: generated/

   LeakyIntegrator
   LIF
   ExpIF
   AdExIF
   QuaIF
   AdQuaIF
   GIF
   Izhikevich
   HindmarshRose
   FHN
   ALIFBellec2020
   LIF_SFA_Bellec2020

