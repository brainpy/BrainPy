``brainpy.connect`` module
==========================

.. currentmodule:: brainpy.connect 
.. automodule:: brainpy.connect 

.. contents::
   :local:
   :depth: 1

Base Connection Classes and Tools
---------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   set_default_dtype
   get_idx_type
   mat2coo
   mat2csc
   mat2csr
   csr2csc
   csr2mat
   csr2coo
   coo2csr
   coo2csc
   coo2mat
   Connector
   TwoEndConnector
   OneEndConnector


Custom Connections
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   MatConn
   IJConn
   CSRConn
   SparseMatConn


Random Connections
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   FixedProb
   FixedPreNum
   FixedPostNum
   FixedTotalNum
   GaussianProb
   ProbDist
   SmallWorld
   ScaleFreeBA
   ScaleFreeBADual
   PowerLaw


Regular Connections
-------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   One2One
   All2All
   GridFour
   GridEight
   GridN
   one2one
   all2all
   grid_four
   grid_eight


