# -*- coding: utf-8 -*-

from typing import Dict, Sequence, Any, Union, Optional

import brainpy.math as bm
from brainpy._src.dynsys import DynamicalSystem
from brainpy._src.runners import DSRunner
from brainpy._src.running import constants as c
from brainpy.errors import NoLongerSupportError
from brainpy.types import ArrayType, Output

__all__ = [
  'DSTrainer',
]


class DSTrainer(DSRunner):
  """Structural Trainer for Dynamical Systems.

  For more parameters, users should refer to :py:class:`~.DSRunner`.

  Parameters
  ----------
  target: DynamicalSystem
    The training target.

  kwargs: Any
    Other general parameters in :py:class:`~.DSRunner`.

  """

  target: DynamicalSystem
  '''The training target.'''

  train_nodes: Sequence[DynamicalSystem]  # need to be initialized by subclass
  '''All children nodes in this training target.'''

  def __init__(
      self,
      target: DynamicalSystem,
      **kwargs
  ):
    super().__init__(target=target, **kwargs)

    if not isinstance(self.target.mode, bm.BatchingMode):
      raise NoLongerSupportError(f'''
      From version 2.3.1, DSTrainer must receive a DynamicalSystem instance with 
      the computing mode of {bm.batching_mode} or {bm.training_mode}. 
      
      See https://github.com/brainpy/BrainPy/releases/tag/V2.3.1
      for the solution of how to fix this.
      ''')

    # jit
    if isinstance(self._origin_jit, bool):
      self.jit[c.PREDICT_PHASE] = self._origin_jit
      self.jit[c.FIT_PHASE] = self._origin_jit
    else:
      self.jit[c.PREDICT_PHASE] = self._origin_jit.get(c.PREDICT_PHASE, True)
      self.jit[c.FIT_PHASE] = self._origin_jit.get(c.FIT_PHASE, True)

  def predict(
      self,
      inputs: Any,
      reset_state: bool = False,
      shared_args: Optional[Dict] = None,
      eval_time: bool = False
  ) -> Output:
    """Prediction function.

    Parameters
    ----------
    inputs: ArrayType, sequence of ArrayType, dict of ArrayType
      The input values.
    reset_state: bool
      Reset the target state before running.
    eval_time: bool
      Whether we evaluate the running time or not?
    shared_args: dict
      The shared arguments across nodes.

    Returns
    -------
    output: ArrayType, sequence of ArrayType, dict of ArrayType
      The running output.
    """
    if shared_args is None:
      shared_args = dict()
    shared_args['fit'] = shared_args.get('fit', False)
    return super().predict(inputs=inputs,
                           reset_state=reset_state,
                           shared_args=shared_args,
                           eval_time=eval_time)

  def fit(
      self,
      train_data: Any,
      reset_state: bool = False,
      shared_args: Dict = None
  ) -> Output:  # need to be implemented by subclass
    raise NotImplementedError('Must implement the fit function. ')
