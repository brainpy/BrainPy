# -*- coding: utf-8 -*-

"""
This module provides methods to initialize weights.
You can access them through ``brainpy.init.XXX``.
"""


from brainpy._src.initialize.base import (
  Initializer as Initializer,
)

from brainpy._src.initialize.decay_inits import (
  GaussianDecay as GaussianDecay,
  DOGDecay as DOGDecay,
)


from brainpy._src.initialize.random_inits import (
  Normal as Normal,
  Uniform as Uniform,
  VarianceScaling as VarianceScaling,
  KaimingUniform as KaimingUniform,
  KaimingNormal as KaimingNormal,
  XavierUniform as XavierUniform,
  XavierNormal as XavierNormal,
  LecunUniform as LecunUniform,
  LecunNormal as LecunNormal,
  Orthogonal as Orthogonal,
  DeltaOrthogonal as DeltaOrthogonal,
  Gamma,
  Exponential,
)


from brainpy._src.initialize.regular_inits import (
  ZeroInit as ZeroInit,
  Constant as Constant,
  OneInit as OneInit,
  Identity as Identity,
)


from brainpy._src.initialize.generic import (
  parameter as parameter,
  variable as variable,
  variable_ as variable_,
  noise as noise,
  delay as delay,
)

from brainpy._src.initialize.others import (
  Clip as Clip,
)

