import numbers
import warnings
from typing import Union, Optional

from brainpy import check, math as bm
from brainpy._src.context import share
from brainpy._src.dynsys import Projection

__all__ = [
  'PoissonInput',
]


class PoissonInput(Projection):
  """Poisson Input to the given :py:class:`~.Variable`.

  Adds independent Poisson input to a target variable. For large
  numbers of inputs, this is much more efficient than creating a
  `PoissonGroup`. The synaptic events are generated randomly during the
  simulation and are not preloaded and stored in memory. All the inputs must
  target the same variable, have the same frequency and same synaptic weight.
  All neurons in the target variable receive independent realizations of
  Poisson spike trains.

  Args:
    target_var: The variable that is targeted by this input. Should be an instance of :py:class:`~.Variable`.
    num_input: The number of inputs.
    freq: The frequency of each of the inputs. Must be a scalar.
    weight: The synaptic weight. Must be a scalar.
    name: The target name.
    mode: The computing mode.
  """

  def __init__(
      self,
      target_var: bm.Variable,
      num_input: int,
      freq: Union[int, float],
      weight: Union[int, float],
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
      seed=None
  ):
    super().__init__(name=name, mode=mode)

    if seed is not None:
      warnings.warn('')

    if not isinstance(target_var, bm.Variable):
      raise TypeError(f'"target_var" must be an instance of Variable. '
                      f'But we got {type(target_var)}: {target_var}')
    self.target_var = target_var
    self.num_input = check.is_integer(num_input, min_bound=1)
    self.freq = check.is_float(freq, min_bound=0., allow_int=True)
    self.weight = check.is_float(weight, allow_int=True)

  def update(self):
    p = self.freq * share['dt'] / 1e3
    a = self.num_input * p
    b = self.num_input * (1 - p)

    if isinstance(share['dt'], numbers.Number):  # dt is not traced
      if (a > 5) and (b > 5):
        inp = bm.random.normal(a, b * p, self.target_var.shape)
      else:
        inp = bm.random.binomial(self.num_input, p, self.target_var.shape)

    else:  # dt is traced
      inp = bm.cond((a > 5) * (b > 5),
                    lambda: bm.random.normal(a, b * p, self.target_var.shape),
                    lambda: bm.random.binomial(self.num_input, p, self.target_var.shape),
                    ())

    # inp = bm.sharding.partition(inp, self.target_var.sharding)
    self.target_var += inp * self.weight

  def __repr__(self):
    return f'{self.name}(num_input={self.num_input}, freq={self.freq}, weight={self.weight})'
