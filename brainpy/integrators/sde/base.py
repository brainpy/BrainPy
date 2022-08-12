# -*- coding: utf-8 -*-

from typing import Dict, Callable, Union, Sequence

from brainpy import math as bm, errors
from brainpy.integrators import constants, utils
from brainpy.integrators.base import Integrator

__all__ = [
  'SDEIntegrator',
]


def f_names(f):
  func_name = constants.unique_name('sde')
  if f.__name__.isidentifier():
    func_name += '_' + f.__name__
  return func_name


class SDEIntegrator(Integrator):
  """SDE Integrator."""

  def __init__(
      self,
      f: Callable,
      g: Callable,
      dt: float = None,
      name: str = None,
      show_code: bool = False,
      var_type: str = None,
      intg_type: str = None,
      wiener_type: str = None,
      state_delays: Dict[str, bm.AbstractDelay] = None,
      dyn_vars: Union[bm.Variable, Sequence[bm.Variable], Dict[str, bm.Variable]] = None,
  ):
    self.dyn_vars = dyn_vars
    dt = bm.get_dt() if dt is None else dt
    parses = utils.get_args(f)
    variables = parses[0]  # variable names, (before 't')
    parameters = parses[1]  # parameter names, (after 't')
    arguments = parses[2]  # function arguments

    # super initialization
    super(SDEIntegrator, self).__init__(name=name,
                                        variables=variables,
                                        parameters=parameters,
                                        arguments=arguments,
                                        dt=dt,
                                        state_delays=state_delays)

    # derivative functions
    self.derivative = {constants.F: f, constants.G: g}
    self.f = f
    self.g = g

    # essential parameters
    intg_type = constants.ITO_SDE if intg_type is None else intg_type
    var_type = constants.SCALAR_VAR if var_type is None else var_type
    wiener_type = constants.SCALAR_WIENER if wiener_type is None else wiener_type
    if intg_type not in constants.SUPPORTED_INTG_TYPE:
      raise errors.IntegratorError(f'Currently, BrainPy only support SDE_INT types: '
                                   f'{constants.SUPPORTED_INTG_TYPE}. But we got {intg_type}.')
    if var_type not in constants.SUPPORTED_VAR_TYPE:
      raise errors.IntegratorError(f'Currently, BrainPy only supports variable types: '
                                   f'{constants.SUPPORTED_VAR_TYPE}. But we got {var_type}.')
    if wiener_type not in constants.SUPPORTED_WIENER_TYPE:
      raise errors.IntegratorError(f'Currently, BrainPy only supports Wiener '
                                   f'Process types: {constants.SUPPORTED_WIENER_TYPE}. '
                                   f'But we got {wiener_type}.')
    self.var_type = var_type  # variable type
    self.intg_type = intg_type  # integral type
    self.wiener_type = wiener_type  # wiener process type

    # random seed
    self.rng = bm.random.RandomState()

    # code scope
    self.code_scope = {constants.F: f, constants.G: g, 'math': bm, 'random': self.rng}
    # code lines
    self.func_name = f_names(f)
    self.code_lines = [f'def {self.func_name}({", ".join(self.arguments)}):']
    # others
    self.show_code = show_code

  def _check_vector_wiener_dim(self, noise_size, var_size):
    if noise_size[:-1] > var_size[-len(noise_size) +1:]:
      raise ValueError(f"Incompatible shapes for shapes of noise {noise_size} and variable {var_size}")
