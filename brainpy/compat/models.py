# -*- coding: utf-8 -*-

import warnings

from brainpy.dyn import neurons, synapses

__all__ = [
  'LIF',
  'AdExIF',
  'Izhikevich',
  'ExpCOBA',
  'ExpCUBA',
  'DeltaSynapse',
]


class LIF(neurons.LIF):
  """LIF neuron model.

  .. deprecated:: 2.1.0
     Please use "brainpy.dyn.LIF" instead.
  """

  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.dyn.LIF" instead. '
                  '"brainpy.models.LIF" is deprecated since '
                  'version 2.1.0', DeprecationWarning)
    super(LIF, self).__init__(*args, **kwargs)


class AdExIF(neurons.AdExIF):
  """AdExIF neuron model.

  .. deprecated:: 2.1.0
     Please use "brainpy.dyn.AdExIF" instead.
  """

  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.dyn.AdExIF" instead. '
                  '"brainpy.models.AdExIF" is deprecated since '
                  'version 2.1.0', DeprecationWarning)
    super(AdExIF, self).__init__(*args, **kwargs)


class Izhikevich(neurons.Izhikevich):
  """Izhikevich neuron model.

  .. deprecated:: 2.1.0
     Please use "brainpy.dyn.Izhikevich" instead.
  """

  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.dyn.Izhikevich" instead. '
                  '"brainpy.models.Izhikevich" is deprecated since '
                  'version 2.1.0', DeprecationWarning)
    super(Izhikevich, self).__init__(*args, **kwargs)


class ExpCOBA(synapses.ExpCOBA):
  """ExpCOBA synapse model.

  .. deprecated:: 2.1.0
     Please use "brainpy.dyn.ExpCOBA" instead.
  """

  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.dyn.ExpCOBA" instead. '
                  '"brainpy.models.ExpCOBA" is deprecated since '
                  'version 2.1.0', DeprecationWarning)
    super(ExpCOBA, self).__init__(*args, **kwargs)


class ExpCUBA(synapses.ExpCUBA):
  """ExpCUBA synapse model.

  .. deprecated:: 2.1.0
     Please use "brainpy.dyn.ExpCUBA" instead.
  """

  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.dyn.ExpCUBA" instead. '
                  '"brainpy.models.ExpCUBA" is deprecated since '
                  'version 2.1.0', DeprecationWarning)
    super(ExpCUBA, self).__init__(*args, **kwargs)


class DeltaSynapse(synapses.DeltaSynapse):
  """Delta synapse model.

  .. deprecated:: 2.1.0
     Please use "brainpy.dyn.DeltaSynapse" instead.
  """

  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.dyn.DeltaSynapse" instead. '
                  '"brainpy.models.DeltaSynapse" is deprecated since '
                  'version 2.1.0', DeprecationWarning)
    super(DeltaSynapse, self).__init__(*args, **kwargs)
