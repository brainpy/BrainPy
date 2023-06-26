# -*- coding: utf-8 -*-


from brainpy._src.dnn.base import (
  Layer as Layer,
)


from brainpy._src.dnn.conv import (
  Conv1d as Conv1d,
  Conv2d as Conv2d,
  Conv3d as Conv3d,
  Conv1D as Conv1D,
  Conv2D as Conv2D,
  Conv3D as Conv3D,
  ConvTranspose1d as ConvTranspose1d,
  ConvTranspose2d as ConvTranspose2d,
  ConvTranspose3d as ConvTranspose3d,
)


from brainpy._src.dnn.dropout import (
  Dropout as Dropout,
)


from brainpy._src.dnn.function import (
  Activation as Activation,
  Flatten as Flatten,
  FunAsLayer as FunAsLayer,
)


from brainpy._src.dnn.linear import (
  Dense as Dense,
  Linear as Linear,
  Identity as Identity,
  AllToAll as AllToAll,
  OneToOne as OneToOne,
  MaskedLinear as MaskedLinear,
  CSRLinear as CSRLinear,
  EventCSRLinear as EventCSRLinear,
  JitFPHomoLinear as JitFPHomoLinear,
  JitFPUniformLinear as JitFPUniformLinear,
  JitFPNormalLinear as JitFPNormalLinear,
  EventJitFPHomoLinear as EventJitFPHomoLinear,
  EventJitFPNormalLinear as EventJitFPNormalLinear,
  EventJitFPUniformLinear as EventJitFPUniformLinear,
)


from brainpy._src.dnn.normalization import (
  BatchNorm1d as BatchNorm1d,
  BatchNorm2d as BatchNorm2d,
  BatchNorm3d as BatchNorm3d,
  BatchNorm1D as BatchNorm1D,
  BatchNorm2D as BatchNorm2D,
  BatchNorm3D as BatchNorm3D,
  LayerNorm as LayerNorm,
  GroupNorm as GroupNorm,
  InstanceNorm as InstanceNorm,
)


from brainpy._src.dnn.nvar import (
  NVAR as NVAR,
)


from brainpy._src.dnn.pooling import (
  MaxPool as MaxPool,
  MaxPool1d as MaxPool1d,
  MaxPool2d as MaxPool2d,
  MaxPool3d as MaxPool3d,

  MinPool as MinPool,

  AvgPool as AvgPool,
  AvgPool1d as AvgPool1d,
  AvgPool2d as AvgPool2d,
  AvgPool3d as AvgPool3d,

  AdaptiveAvgPool1d as AdaptiveAvgPool1d,
  AdaptiveAvgPool2d as AdaptiveAvgPool2d,
  AdaptiveAvgPool3d as AdaptiveAvgPool3d,
  AdaptiveMaxPool1d as AdaptiveMaxPool1d,
  AdaptiveMaxPool2d as AdaptiveMaxPool2d,
  AdaptiveMaxPool3d as AdaptiveMaxPool3d,
)


from brainpy._src.dnn.reservoir import (
  Reservoir as Reservoir,
)


from brainpy._src.dnn.rnncells import (
  RNNCell as RNNCell,
  GRUCell as GRUCell,
  LSTMCell as LSTMCell,
  Conv1dLSTMCell as Conv1dLSTMCell,
  Conv2dLSTMCell as Conv2dLSTMCell,
  Conv3dLSTMCell as Conv3dLSTMCell,
)


from brainpy._src.dnn.interoperation_flax import (
  FromFlax,
  ToFlaxRNNCell, ToFlax,
)


from brainpy._src.dnn.activations import (
  Threshold,
  ReLU,
  RReLU,
  Hardtanh,
  ReLU6,
  Sigmoid,
  Hardsigmoid,
  Tanh,
  SiLU,
  Mish,
  Hardswish,
  ELU,
  CELU,
  SELU,
  GLU,
  GELU,
  Hardshrink,
  LeakyReLU,
  LogSigmoid,
  Softplus,
  Softshrink,
  PReLU,
  Softsign,
  Tanhshrink,
  Softmin,
  Softmax,
  Softmax2d,
  LogSoftmax,
)

