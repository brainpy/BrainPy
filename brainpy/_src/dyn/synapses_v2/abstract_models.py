# -*- coding: utf-8 -*-

from typing import Union, Dict, Callable, Optional

from jax import vmap

import brainpy.math as bm
from brainpy._src import tools
from brainpy._src.connect import TwoEndConnector, All2All, One2One
from brainpy._src.dyn.context import share
from brainpy._src.dyn.synapses_v2.base import SynConn, SynOut, SynSTP
from brainpy._src.initialize import Initializer, variable_
from brainpy._src.integrators import odeint
from brainpy.check import is_float
from brainpy.types import ArrayType


class Exponential(SynConn):
  r"""Exponential decay synapse model.

  **Model Descriptions**

  The single exponential decay synapse model assumes the release of neurotransmitter,
  its diffusion across the cleft, the receptor binding, and channel opening all happen
  very quickly, so that the channels instantaneously jump from the closed to the open state.
  Therefore, its expression is given by

  .. math::

      g_{\mathrm{syn}}(t)=g_{\mathrm{max}} e^{-\left(t-t_{0}\right) / \tau}

  where :math:`\tau_{delay}` is the time constant of the synaptic state decay,
  :math:`t_0` is the time of the pre-synaptic spike,
  :math:`g_{\mathrm{max}}` is the maximal conductance.

  Accordingly, the differential form of the exponential synapse is given by

  .. math::

      \begin{aligned}
       & g_{\mathrm{syn}}(t) = g_{max} g * \mathrm{STP} \\
       & \frac{d g}{d t} = -\frac{g}{\tau_{decay}}+\sum_{k} \delta(t-t_{j}^{k}).
       \end{aligned}

  where :math:`\mathrm{STP}` is used to model the short-term plasticity effect.

  Parameters
  ----------
  conn: optional, ArrayType, dict of (str, ndarray), TwoEndConnector
    The synaptic connections.
  comp_method: str
    The connection type used for model speed optimization. It can be
    `sparse` and `dense`. The default is `sparse`.
  tau: float, ArrayType
    The time constant of decay. [ms]
  g_max: float, ArrayType, Initializer, Callable
    The synaptic strength (the maximum conductance). Default is 1.
  name: str
    The name of this synaptic projection.
  method: str
    The numerical integration methods.

  References
  ----------

  .. [1] Sterratt, David, Bruce Graham, Andrew Gillies, and David Willshaw.
          "The Synapse." Principles of Computational Modelling in Neuroscience.
          Cambridge: Cambridge UP, 2011. 172-95. Print.

  """

  def __init__(
      self,
      conn: Union[TwoEndConnector, ArrayType, Dict[str, ArrayType]],
      out: Optional[SynOut] = None,
      stp: Optional[SynSTP] = None,
      comp_method: str = 'sparse',
      g_max: Union[float, ArrayType, Initializer, Callable] = 1.,
      tau: Union[float, ArrayType] = 8.0,
      method: str = 'exp_auto',
      name: str = None,
      mode: bm.Mode = None,
  ):
    super().__init__(conn=conn,
                     out=out,
                     stp=stp,
                     name=name,
                     mode=mode)

    # parameters
    self.comp_method = comp_method
    self.tau = is_float(tau, allow_int=True)

    # connections and weights
    self.g_max, self.conn_mask = self._init_weights(g_max, comp_method, data_if_sparse='csr')

    # function
    self.integral = odeint(lambda g, t: -g / self.tau, method=method)

    # variables
    self.reset_state(self.mode)

  def reset_state(self, batch_size=None):
    self.g = variable_(bm.zeros, self.post_num, batch_size)
    if self.out is not None:
      self.out.reset_state(batch_size)
    if self.stp is not None:
      self.stp.reset_state(batch_size)

  def update(self, pre_spike):
    if self.stp is not None:
      syn_value = self.stp(pre_spike) * pre_spike
    else:
      syn_value = bm.asarray(pre_spike, dtype=bm.float_)

    # post values
    if isinstance(self.conn, All2All):
      post_vs = self._syn2post_with_all2all(syn_value, self.g_max, self.conn.include_self)
    elif isinstance(self.conn, One2One):
      post_vs = self._syn2post_with_one2one(syn_value, self.g_max)
    else:
      if self.comp_method == 'sparse':
        bl = tools.import_brainpylib()

        if self.stp is None:
          f = lambda s: bl.event_ops.event_csr_matvec(self.g_max,
                                                      self.conn_mask[0],
                                                      self.conn_mask[1],
                                                      s,
                                                      shape=(self.pre_num, self.post_num),
                                                      transpose=True)
          if isinstance(self.mode, bm.BatchingMode):
            f = vmap(f)
          post_vs = f(pre_spike)
        else:
          f = lambda s: bl.sparse_ops.cusparse_csr_matvec(self.g_max,
                                                          self.conn_mask[0],
                                                          self.conn_mask[1],
                                                          s,
                                                          shape=(self.pre_num, self.post_num),
                                                          transpose=True)
          if isinstance(self.mode, bm.BatchingMode):
            f = vmap(f)
          post_vs = f(syn_value)
      else:
        post_vs = self._syn2post_with_dense(syn_value, self.g_max, self.conn_mask)

    # updates
    self.g.value = self.integral(self.g.value, share.load('t'), bm.dt) + post_vs

    # outputs
    if self.out is not None:
      return self.out(self.g.value)
    else:
      return self.g.value
