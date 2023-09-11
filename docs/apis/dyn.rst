``brainpy.dyn`` module
======================

.. currentmodule:: brainpy.dyn 
.. automodule:: brainpy.dyn 

.. contents::
   :local:
   :depth: 1

Base Classes
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   NeuDyn
   SynDyn
   IonChaDyn


Ion Dynamics
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   mix_ions
   Ion
   MixIons
   Calcium
   CalciumFixed
   CalciumDetailed
   CalciumFirstOrder
   Sodium
   SodiumFixed
   Potassium
   PotassiumFixed


Ion Channel Dynamics
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   IonChannel
   CalciumChannel
   ICaN_IS2008
   ICaT_HM1992
   ICaT_HP1992
   ICaHT_HM1992
   ICaHT_Re1993
   ICaL_IS2008
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
   Ih_HM1992
   Ih_De1996
   IAHP_De1994v2
   IAHP_De1994
   SodiumChannel
   INa_Ba2002
   INa_TM1991
   INa_HH1952
   INa_Ba2002v2
   INa_TM1991v2
   INa_HH1952v2
   LeakyChannel
   IL


Neuron Dynamics
---------------

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


Synaptic Dynamics
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Delta
   Expon
   Alpha
   DualExpon
   DualExponV2
   NMDA
   STD
   STP
   AMPA
   GABAa
   BioNMDA
   DiffusiveCoupling
   AdditiveCoupling


Synaptic Projections
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   VanillaProj
   ProjAlignPostMg1
   ProjAlignPostMg2
   ProjAlignPost1
   ProjAlignPost2
   ProjAlignPreMg1
   ProjAlignPreMg2
   ProjAlignPre1
   ProjAlignPre2
   SynConn
   PoissonInput


Common Dynamical Models
-----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Leaky
   Integrator
   InputGroup
   OutputGroup
   SpikeTimeGroup
   PoissonGroup
   OUProcess


Synaptic Output Models
----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   SynOut
   COBA
   CUBA
   MgBlock


Population Rate Models
----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   FHN
   FeedbackFHN
   QIF
   StuartLandauOscillator
   WilsonCowanModel
   ThresholdLinearModel


