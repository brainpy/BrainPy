``brainpy.dnn`` module
======================

.. currentmodule:: brainpy.dnn 
.. automodule:: brainpy.dnn 

.. contents::
   :local:
   :depth: 1

Non-linear Activations
----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Activation
   Threshold
   ReLU
   RReLU
   Hardtanh
   ReLU6
   Sigmoid
   Hardsigmoid
   Tanh
   SiLU
   Mish
   Hardswish
   ELU
   CELU
   SELU
   GLU
   GELU
   Hardshrink
   LeakyReLU
   LogSigmoid
   Softplus
   Softshrink
   PReLU
   Softsign
   Tanhshrink
   Softmin
   Softmax
   Softmax2d
   LogSoftmax


Convolutional Layers
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Conv1d
   Conv2d
   Conv3d
   Conv1D
   Conv2D
   Conv3D
   ConvTranspose1d
   ConvTranspose2d
   ConvTranspose3d


Dense Connection Layers
-----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Dense
   Linear
   Identity
   AllToAll
   OneToOne
   MaskedLinear
   CSRLinear
   EventCSRLinear
   JitFPHomoLinear
   JitFPUniformLinear
   JitFPNormalLinear
   EventJitFPHomoLinear
   EventJitFPNormalLinear
   EventJitFPUniformLinear


Normalization Layers
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   BatchNorm1d
   BatchNorm2d
   BatchNorm3d
   BatchNorm1D
   BatchNorm2D
   BatchNorm3D
   LayerNorm
   GroupNorm
   InstanceNorm


Pooling Layers
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   MaxPool
   MaxPool1d
   MaxPool2d
   MaxPool3d
   MinPool
   AvgPool
   AvgPool1d
   AvgPool2d
   AvgPool3d
   AdaptiveAvgPool1d
   AdaptiveAvgPool2d
   AdaptiveAvgPool3d
   AdaptiveMaxPool1d
   AdaptiveMaxPool2d
   AdaptiveMaxPool3d



Interoperation with Flax
------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   FromFlax
   ToFlaxRNNCell
   ToFlax


Utility Layers
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Dropout
   Flatten
   Unflatten
   FunAsLayer

