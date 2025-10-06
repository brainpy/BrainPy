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


class GPUOperatorNotFound(Exception):
    __module__ = 'brainpy'

    def __init__(self, name):
        super(GPUOperatorNotFound, self).__init__(f'''
GPU operator for "{name}" does not found. 

Please install brainpylib GPU operators with linux + CUDA environment.
    ''')


class SharedArgError(BrainPyError):
    __module__ = 'brainpy'
