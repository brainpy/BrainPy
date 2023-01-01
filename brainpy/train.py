# -*- coding: utf-8 -*-

from brainpy._src.train.base import (
  DSTrainer as DSTrainer
)

from brainpy._src.train.back_propagation import (
  BPTT as BPTT,
  BPFF as BPFF,
)

from brainpy._src.train.online import (
  OnlineTrainer as OnlineTrainer,
  ForceTrainer as ForceTrainer,
)

from brainpy._src.train.offline import (
  OfflineTrainer as OfflineTrainer,
  RidgeTrainer as RidgeTrainer,
)
