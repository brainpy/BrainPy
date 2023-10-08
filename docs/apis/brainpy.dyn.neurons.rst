Neuron Dynamics
===============

.. currentmodule:: brainpy.dyn
.. automodule:: brainpy.dyn

Neuronal dynamics refers to the diverse temporal activity patterns exhibited by neurons related to how they process and transmit information. Key aspects include:

- Action potential generation - the dynamics of neurons rapidly depolarizing and firing action potentials, often in spatially propagating waves along axons. This forms the basis of neural signaling.
- Refractoriness - the brief period after an action potential when a neuron is less excitable and cannot fire again. This leads to limits on maximal firing rates.
- Spiking patterns - neurons can exhibit various firing patterns like tonic regular spiking, bursting, adaptation, etc. This depends on their intrinsic properties.
- Subthreshold activity - fluctuations, oscillations and waves of membrane potential below threshold for spiking, reflecting integrative processes in dendrites.
- Resonance - some neurons exhibit resonance at certain preferred input frequencies, selectively amplifying inputs at that band.
- Spike frequency adaptation - the rate of neuronal firing adapts or decreases in response to sustained input due to intrinsic currents.
- Bistability - some neurons can switch between two stable resting potentials, acting like a toggle switch.
- Excitability - changes in neuron excitability mediated by neuromodulators, activity history, etc. This alters how inputs are translated into outputs.
- Stochasticity - randomness in ion channel openings/closings and synaptic transmission causes neuronal activity to exhibit inherent variability.
- Homeostatic regulation - maintaining firing rates within a stable range by globally adapting ion channel properties and excitability.

So in summary, neuronal dynamics describe the rich temporal patterns of electrical signaling exhibited by neurons, which ultimately underlies how information is encoded and processed in the brain.


Reduced Neuron Models
---------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Lif
   LifLTC
   LifRefLTC
   LifRef
   ExpIF
   ExpIFLTC
   ExpIFRefLTC
   ExpIFRef
   AdExIF
   AdExIFLTC
   AdExIFRefLTC
   AdExIFRef
   QuaIF
   QuaIFLTC
   QuaIFRefLTC
   QuaIFRef
   AdQuaIF
   AdQuaIFLTC
   AdQuaIFRefLTC
   AdQuaIFRef
   Gif
   GifLTC
   GifRefLTC
   GifRef
   Izhikevich
   IzhikevichLTC
   IzhikevichRefLTC
   IzhikevichRef
   HHTypedNeuron
   CondNeuGroupLTC
   CondNeuGroup
   HH
   HHLTC
   MorrisLecar
   MorrisLecarLTC
   WangBuzsakiHH
   WangBuzsakiHHLTC


Hodgkin–Huxley Neuron Models
----------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   HHTypedNeuron
   CondNeuGroupLTC
   CondNeuGroup
   HH
   HHLTC
   MorrisLecar
   MorrisLecarLTC
   WangBuzsakiHH
   WangBuzsakiHHLTC

