# -*- coding: utf-8 -*-


from brainpy._src.dyn.layers.base import (
  Layer as Layer,
)

from brainpy._src.dyn.layers.conv import (
  Conv1d as Conv1d,
  Conv2d as Conv2d,
  Conv3d as Conv3d,
)
Conv1D = Conv1d
Conv2D = Conv2d
Conv3D = Conv3d

from brainpy._src.dyn.layers.dropout import (
  Dropout as Dropout,
)

from brainpy._src.dyn.layers.function import (
  Activation as Activation,
  Flatten as Flatten,
  FunAsLayer as FunAsLayer,
)

from brainpy._src.dyn.layers.linear import (
  Dense as Dense,
)

from brainpy._src.dyn.layers.normalization import (
  BatchNorm1d as BatchNorm1d,
  BatchNorm2d as BatchNorm2d,
  BatchNorm3d as BatchNorm3d,
  LayerNorm as LayerNorm,
  GroupNorm as GroupNorm,
  InstanceNorm as InstanceNorm,
)
BatchNorm1D = BatchNorm1d
BatchNorm2D = BatchNorm2d
BatchNorm3D = BatchNorm3d


from brainpy._src.dyn.layers.nvar import (
  NVAR as NVAR,
)

from brainpy._src.dyn.layers.pooling import (
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

from brainpy._src.dyn.layers.reservoir import (
  Reservoir as Reservoir,
)

from brainpy._src.dyn.layers.rnncells import (
  RNNCell as RNNCell,
  GRUCell as GRUCell,
  LSTMCell as LSTMCell,
  VanillaRNN as VanillaRNN,
  GRU as GRU,
  LSTM as LSTM,
)
