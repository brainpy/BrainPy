# -*- coding: utf-8 -*-


class BrainPyError(Exception):
  """General BrainPy error."""
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


class MPACheckpointingRequiredError(BrainPyError):
  """To optimally save and restore a multiprocess array (GDA or jax Array outputted from pjit), use GlobalAsyncCheckpointManager.

  You can create an GlobalAsyncCheckpointManager at top-level and pass it as
  argument::

    from jax.experimental.gda_serialization import serialization as gdas
    gda_manager = gdas.GlobalAsyncCheckpointManager()
    brainpy.checkpoints.save(..., gda_manager=gda_manager)
  """

  def __init__(self, path, step):
    super().__init__(
      f'Checkpoint failed at step: "{step}" and path: "{path}": Target '
      'contains a multiprocess array should be saved/restored with a '
      'GlobalAsyncCheckpointManager.')


class MPARestoreTargetRequiredError(BrainPyError):
  """Provide a valid target when restoring a checkpoint with a multiprocess array.

  Multiprocess arrays need a sharding (global meshes and partition specs) to be
  initialized. Therefore, to restore a checkpoint that contains a multiprocess
  array, make sure the ``target`` you passed contains valid multiprocess arrays
  at the corresponding tree structure location. If you cannot provide a full
  valid ``target``, consider ``allow_partial_mpa_restoration=True``.
  """

  def __init__(self, path, step, key=None):
    error_msg = (
      f'Restore checkpoint failed at step: "{step}" and path: "{path}": '
      'Checkpoints containing a multiprocess array need to be restored with '
      'a target with pre-created arrays. If you cannot provide a full valid '
      'target, consider ``allow_partial_mpa_restoration=True``. ')
    if key:
      error_msg += f'This error fired when trying to restore array at {key}.'
    super().__init__(error_msg)


class MPARestoreDataCorruptedError(BrainPyError):
  """A multiprocess array stored in Google Cloud Storage doesn't contain a "commit_success.txt" file, which should be written at the end of the save.

  Failure of finding it could indicate a corruption of your saved GDA data.
  """

  def __init__(self, step, path):
    super().__init__(
      f'Restore checkpoint failed at step: "{step}" on multiprocess array at '
      f' "{path}": No "commit_success.txt" found on this "_gda" directory. '
      'Was its save halted before completion?')


class MPARestoreTypeNotMatchError(BrainPyError):
  """Make sure the multiprocess array type you use matches your configuration in jax.config.jax_array.

  If you turned `jax.config.jax_array` on, you should use
  `jax.experimental.array.Array` everywhere, instead of using
  `GlobalDeviceArray`. Otherwise, avoid using jax.experimental.array
  to restore your checkpoint.
  """

  def __init__(self, step, gda_path):
    super().__init__(
      f'Restore checkpoint failed at step: "{step}" on multiprocess array at '
      f' "{gda_path}": The array type provided by the target does not match '
      'the JAX global configuration, namely the jax.config.jax_array.')


class AlreadyExistsError(BrainPyError):
  """Attempting to overwrite a file via copy.

  You can pass ``overwrite=True`` to disable this behavior and overwrite
  existing files in.
  """

  def __init__(self, path):
    super().__init__(f'Trying overwrite an existing file: "{path}".')


class InvalidCheckpointError(BrainPyError):
  """A checkpoint cannot be stored in a directory that already has

  a checkpoint at the current or a later step.

  You can pass ``overwrite=True`` to disable this behavior and
  overwrite existing checkpoints in the target directory.
  """

  def __init__(self, path, step):
    super().__init__(
      f'Trying to save an outdated checkpoint at step: "{step}" and path: "{path}".'
    )


class InvalidCheckpointPath(BrainPyError):
  """A checkpoint cannot be stored in a directory that already has

  a checkpoint at the current or a later step.

  You can pass ``overwrite=True`` to disable this behavior and
  overwrite existing checkpoints in the target directory.
  """

  def __init__(self, path):
    super().__init__(f'Invalid checkpoint at "{path}".')


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


class GPUOperatorNotFound(Exception):
  def __init__(self, name):
    super(GPUOperatorNotFound, self).__init__(f'''
GPU operator for "{name}" does not found. 

Please install brainpylib GPU operators with linux + CUDA environment.
    ''')




class SharedArgError(BrainPyError):
  pass


