# -*- coding: utf-8 -*-


class BrainPyError(Exception):
  """General BrainPy error."""
  pass


class ModelBuildError(BrainPyError):
  """The error occurred during the definition of models."""
  pass


class RunningError(BrainPyError):
  """The error occurred in the running function."""
  pass


class IntegratorError(BrainPyError):
  pass


class DiffEqError(BrainPyError):
  """The differential equation definition error."""
  pass


class CodeError(BrainPyError):
  """Code definition error.
  """
  pass


class AnalyzerError(BrainPyError):
  """Error occurred in differential equation analyzer and dynamics analysis.
  """


class PackageMissingError(BrainPyError):
  """The package missing error.
  """
  pass


class BackendNotInstalled(BrainPyError):
  def __init__(self, backend):
    super(BackendNotInstalled, self).__init__(
      '"{bk}" must be installed when the user wants to use {bk} backend. \n'
      'Please install {bk} through "pip install {bk}" '
      'or "conda install {bk}".'.format(bk=backend))


class UniqueNameError(BrainPyError):
  def __init__(self, *args):
    super(UniqueNameError, self).__init__(*args)


class UnsupportedError(BrainPyError):
  pass


class NoImplementationError(BrainPyError):
  pass


class NoLongerSupportError(BrainPyError):
  pass


class ConnectorError(BrainPyError):
  pass


class MonitorError(BrainPyError):
  pass


class MathError(BrainPyError):
  """Errors occurred in ``brainpy.math`` module."""
  pass


class JaxTracerError(MathError):
  def __init__(self, variables=None):
    msg = 'There is an unexpected tracer. \n\n' \
          'In BrainPy, all the dynamically changed variables must be declared as ' \
          '"brainpy.math.Variable" and they should be provided ' \
          'into the "dyn_vars" when calling the transformation functions, ' \
          'like "jit()", "vmap()", "grad()", "make_loop()", etc. \n\n'

    if variables is None:
      pass
    elif isinstance(variables, dict):
      msg += f'We detect all the provided dynamical variables are: ' \
             f'{variables.keys()}\n\n'
    elif isinstance(variables, (list, tuple)):
      msg += 'We detect all the provided dynamical variables are: \n'
      for v in variables:
        msg += f'\t{v.dtype}[{v.shape}]\n'
      msg += '\n'
    else:
      raise ValueError

    # msg += 'While there are changed variables which are not wrapped into "dyn_vars". Please check!'
    msg = 'While there are changed variables which are not wrapped into "dyn_vars". Please check!'

    super(JaxTracerError, self).__init__(msg)


class ConcretizationTypeError(Exception):
  def __init__(self):
    super(ConcretizationTypeError, self).__init__(
      'This problem may be caused by several ways:\n'
      '1. Your if-else conditional statement relies on instances of brainpy.math.Variable. \n'
      '2. Your if-else conditional statement relies on functional arguments which do not '
      'set in "static_argnames" when applying JIT compilation. More details please see '
      'https://jax.readthedocs.io/en/latest/errors.html#jax.errors.ConcretizationTypeError\n'
      '3. The static variables which set in the "static_argnames" are provided '
      'as arguments, not keyword arguments, like "jit_f(v1, v2)" [<- wrong]. '
      'Please write it as "jit_f(static_k1=v1, static_k2=v2)" [<- right].'
    )

