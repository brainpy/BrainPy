Synaptic Dynamics
======================

.. currentmodule:: brainpy.dyn
.. automodule:: brainpy.dyn


Synaptic dynamics refers to the processes that modulate the strength of synaptic connections between
neurons in the brain. Key aspects of synaptic dynamics include:

- Synaptic plasticity - the ability of synaptic strength to change in response to neuronal activity.
  This includes phenomena like long-term potentiation (LTP) and long-term depression (LTD) which
  increase or decrease synaptic strength based on patterns of activity. These processes allow synapses
  to strengthen or weaken over time and are important for learning and memory.
- Short-term synaptic plasticity - rapid, short-lived changes in synaptic strength in response to
  recent activity patterns. This includes short-term facilitation and depression. These transient
  changes allow synapses to exhibit short-term memory and alter their function on the timescale of
  milliseconds to minutes.
- Homeostatic plasticity - slower compensatory mechanisms that maintain optimal overall activity
  levels in neurons and networks by globally scaling up or down synaptic strengths.
  This stabilizes network function.
- Neuromodulation - the ability of neuromodulators like dopamine, acetylcholine and serotonin to
  alter synaptic transmission, often by changing synaptic plasticity. This allows global shifts
  in network dynamics and learning rules.
- Synaptic noise - fluctuations in synaptic transmission due to the stochastic nature of
  neurotransmitter release. This variability impacts signal transmission and synaptic plasticity.
- Synaptic integration and dynamics - how incoming signals at many thousands of synapses are
  integrated and processed in individual neurons, altering their computational functions.
  This depends on synaptic dynamics.

So in summary, synaptic dynamics describe the various processes that change the properties of synapses
in the brain over timescales ranging from milliseconds to hours/days. These dynamics ultimately
regulate how information is transmitted and encoded in neural circuits.


Phenomenological synapse models
-------------------------------


.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Expon
   Alpha
   DualExpon
   DualExponV2
   NMDA
   STD
   STP



Biological synapse models
-------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   AMPA
   GABAa
   BioNMDA



Gap junction models
-------------------


.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   DiffusiveCoupling
   AdditiveCoupling


