Ion Channel Dynamics
====================



.. currentmodule:: brainpy.dyn
.. automodule:: brainpy.dyn


.. contents::
   :local:
   :depth: 1


Ion channel models are the building blocks of computational neuron models. Their biological fidelity
is therefore crucial for the interpretation of simulations.

Ion channels in the brain are specialized proteins that are embedded in the cell membranes of neurons.
They act as gatekeepers, regulating the flow of specific ions across the membrane in response to various
signals and stimuli. Ion channels are crucial for generating and controlling electrical signals in neurons.

There are different types of ion channels in the brain, each with specific properties and functions. Some
of the most important types include voltage-gated ion channels, ligand-gated ion channels, and leak channels.
Voltage-gated ion channels open or close in response to changes in the electrical potential across the
membrane. Ligand-gated ion channels open or close when specific molecules, such as neurotransmitters,
bind to them. Leak channels allow a small, continuous flow of ions across the membrane, contributing to
the resting membrane potential.

Modeling the dynamics of ion channels in the brain involves capturing their behavior and interactions
using mathematical models and computer simulations.


Base Classes
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: ion_template.rst

   IonChannel



Calcium Channels
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: ion_template.rst

   CalciumChannel
   ICaN_IS2008
   ICaT_HM1992
   ICaT_HP1992
   ICaHT_HM1992
   ICaHT_Re1993
   ICaL_IS2008


Potassium Channels
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: ion_template.rst

   PotassiumChannel
   IKDR_Ba2002v2
   IK_TM1991v2
   IK_HH1952v2
   IKA1_HM1992v2
   IKA2_HM1992v2
   IKK2A_HM1992v2
   IKK2B_HM1992v2
   IKNI_Ya1989v2
   IK_Leak
   IKDR_Ba2002
   IK_TM1991
   IK_HH1952
   IKA1_HM1992
   IKA2_HM1992
   IKK2A_HM1992
   IKK2B_HM1992
   IKNI_Ya1989
   IKL



Sodium Channels
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: ion_template.rst

   SodiumChannel
   INa_Ba2002
   INa_TM1991
   INa_HH1952
   INa_Ba2002v2
   INa_TM1991v2
   INa_HH1952v2


Other Channels
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: ion_template.rst

   Ih_HM1992
   Ih_De1996
   IAHP_De1994v2
   IAHP_De1994
   LeakyChannel
   IL
