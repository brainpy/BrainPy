#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# -*- coding: utf-8 -*-


__all__ = [
    'BrainPyError',
    'APIChangedError',
    'RunningError',
    'IntegratorError',
    'DiffEqError',
    'CodeError',
    'AnalyzerError',
    'PackageMissingError',
    'BackendNotInstalled',
    'UniqueNameError',
    'UnsupportedError',
    'NoImplementationError',
    'NoLongerSupportError',
    'ConnectorError',
    'MonitorError',
    'MathError',
    'JaxTracerError',
    'ConcretizationTypeError',
    'GPUOperatorNotFound',
    'SharedArgError',
]


class BrainPyError(Exception):
    """General BrainPy error."""
    __module__ = 'brainpy'


class APIChangedError(BrainPyError):
    __module__ = 'brainpy'


class RunningError(BrainPyError):
    """The error occurred in the running function."""
    __module__ = 'brainpy'


class IntegratorError(BrainPyError):
    __module__ = 'brainpy'


class DiffEqError(BrainPyError):
    """The differential equation definition error."""
    __module__ = 'brainpy'


class CodeError(BrainPyError):
    """Code definition error.
    """
    __module__ = 'brainpy'


class AnalyzerError(BrainPyError):
    """Error occurred in differential equation analyzer and dynamics analysis.
    """
    __module__ = 'brainpy'


class PackageMissingError(BrainPyError):
    """The package missing error.
    """
    __module__ = 'brainpy'

    @classmethod
    def by_purpose(cls, name, purpose):
        err = (f'"{name}" must be installed when the user wants to use {purpose}. \n'
               f'Please install through "pip install {name}".')
        return cls(err)


class BackendNotInstalled(BrainPyError):
    __module__ = 'brainpy'

    def __init__(self, backend):
        super(BackendNotInstalled, self).__init__(
            '"{bk}" must be installed when the user wants to use {bk} backend. \n'
            'Please install {bk} through "pip install {bk}" '
            'or "conda install {bk}".'.format(bk=backend))


class UniqueNameError(BrainPyError):
    __module__ = 'brainpy'

    def __init__(self, *args):
        super(UniqueNameError, self).__init__(*args)


class UnsupportedError(BrainPyError):
    __module__ = 'brainpy'


class NoImplementationError(BrainPyError):
    __module__ = 'brainpy'


class NoLongerSupportError(BrainPyError):
    __module__ = 'brainpy'


class ConnectorError(BrainPyError):
    __module__ = 'brainpy'


class MonitorError(BrainPyError):
    __module__ = 'brainpy'


class MathError(BrainPyError):
    """Errors occurred in ``brainpy.math`` module."""
    __module__ = 'brainpy'


class MPACheckpointingRequiredError(BrainPyError):
    """To optimally save and restore a multiprocess array (GDA or jax Array outputted from pjit), use GlobalAsyncCheckpointManager.

    You can create an GlobalAsyncCheckpointManager at top-level and pass it as
    argument::

      from jax.experimental.gda_serialization import serialization as gdas
      gda_manager = gdas.GlobalAsyncCheckpointManager()
      brainpy.checkpoints.save(..., gda_manager=gda_manager)
    """
    __module__ = 'brainpy'

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
    __module__ = 'brainpy'

    def __init__(self, path, step, key=None):
        error_msg = (
            f'Restore checkpoint failed at step: "{step}" and path: "{path}": '
            'Checkpoints containing a multiprocess array need to be restored with '
            'a target with pre-created arrays. If you cannot provide a full valid '
            'target, consider ``allow_partial_mpa_restoration=True``. ')
        if key:
            error_msg += f'This error fired when trying to restore array at {key}.'
        super().__init__(error_msg)


class AlreadyExistsError(BrainPyError):
    """Attempting to overwrite a file via copy.

    You can pass ``overwrite=True`` to disable this behavior and overwrite
    existing files in.
    """
    __module__ = 'brainpy'

    def __init__(self, path):
        super().__init__(f'Trying overwrite an existing file: "{path}".')


class InvalidCheckpointError(BrainPyError):
    """A checkpoint cannot be stored in a directory that already has

    a checkpoint at the current or a later step.

    You can pass ``overwrite=True`` to disable this behavior and
    overwrite existing checkpoints in the target directory.
    """
    __module__ = 'brainpy'

    def __init__(self, path, step):
        super().__init__(
            f'Trying to save an outdated checkpoint at step: "{step}" and path: "{path}".'
        )


class JaxTracerError(MathError):
    __module__ = 'brainpy'

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
    __module__ = 'brainpy'

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
    __module__ = 'brainpy'

    def __init__(self, name):
        super(GPUOperatorNotFound, self).__init__(f'''
GPU operator for "{name}" does not found. 

Please install brainpylib GPU operators with linux + CUDA environment.
    ''')


class SharedArgError(BrainPyError):
    __module__ = 'brainpy'
