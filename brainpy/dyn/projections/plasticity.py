# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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
from typing import Optional, Callable, Union

from brainpy import math as bm, check
from brainpy.delay import register_delay_by_return
from brainpy.dyn.synapses.abstract_models import Expon
from brainpy.dynsys import DynamicalSystem, Projection
from brainpy.mixin import (JointType, ParamDescriber, SupportAutoDelay,
                           BindCondData, AlignPost, SupportSTDP)
from brainpy.types import ArrayType
from .align_post import (align_post_add_bef_update, )
from .align_pre import (align_pre2_add_bef_update, )
from .utils import (_get_return, )

__all__ = [
    'STDP_Song2000',
]


def _init_trace_by_align_pre2(
    target: DynamicalSystem,
    delay: Union[None, int, float],
    syn: ParamDescriber[DynamicalSystem],
):
    """Calculate the trace of the target by reusing the existing connections."""
    check.is_instance(target, DynamicalSystem)
    check.is_instance(syn, ParamDescriber[DynamicalSystem])
    # delay initialization
    delay_cls = register_delay_by_return(target)
    # synapse initialization
    syn = align_pre2_add_bef_update(syn, delay, delay_cls)
    return syn


class STDP_Song2000(Projection):
    r"""Spike-time-dependent plasticity proposed by (Song, et. al, 2000).

    This model filters the synaptic currents according to the variables: :math:`w`.

    .. math::

       I_{syn}^+(t) = I_{syn}^-(t) * w

    where :math:`I_{syn}^-(t)` and :math:`I_{syn}^+(t)` are the synaptic currents before
    and after STDP filtering, :math:`w` measures synaptic efficacy because each time a presynaptic neuron emits a pulse,
    the conductance of the synapse will increase w.

    The dynamics of :math:`w` is governed by the following equation:

    .. math::

        \begin{aligned}
        \frac{dw}{dt} & = & -A_{post}\delta(t-t_{sp}) + A_{pre}\delta(t-t_{sp}), \\
        \frac{dA_{pre}}{dt} & = & -\frac{A_{pre}}{\tau_s} + A_1\delta(t-t_{sp}), \\
        \frac{dA_{post}}{dt} & = & -\frac{A_{post}}{\tau_t} + A_2\delta(t-t_{sp}), \\
        \end{aligned}

    where :math:`t_{sp}` denotes the spike time and :math:`A_1` is the increment
    of :math:`A_{pre}`, :math:`A_2` is the increment of :math:`A_{post}` produced by a spike.

    Here is an example of the usage of this class::

      import brainpy as bp
      import brainpy.math as bm

      class STDPNet(bp.DynamicalSystem):
         def __init__(self, num_pre, num_post):
           super().__init__()
           self.pre = bp.dyn.LifRef(num_pre)
           self.post = bp.dyn.LifRef(num_post)
           self.syn = bp.dyn.STDP_Song2000(
             pre=self.pre,
             delay=1.,
             comm=bp.dnn.EventCSRLinear(bp.conn.FixedProb(1, pre=self.pre.num, post=self.post.num),
                                        weight=bp.init.Uniform(max_val=0.1)),
             syn=bp.dyn.Expon.desc(self.post.varshape, tau=5.),
             out=bp.dyn.COBA.desc(E=0.),
             post=self.post,
             tau_s=16.8,
             tau_t=33.7,
             A1=0.96,
             A2=0.53,
           )

         def update(self, I_pre, I_post):
           self.syn()
           self.pre(I_pre)
           self.post(I_post)
           conductance = self.syn.refs['syn'].g
           Apre = self.syn.refs['pre_trace'].g
           Apost = self.syn.refs['post_trace'].g
           current = self.post.sum_inputs(self.post.V)
           return self.pre.spike, self.post.spike, conductance, Apre, Apost, current, self.syn.comm.weight

      duration = 300.
      I_pre = bp.inputs.section_input([0, 30, 0, 30, 0, 30, 0, 30, 0, 30, 0, 30, 0],
                                      [5, 15, 15, 15, 15, 15, 100, 15, 15, 15, 15, 15, duration - 255])
      I_post = bp.inputs.section_input([0, 30, 0, 30, 0, 30, 0, 30, 0, 30, 0, 30, 0],
                                       [10, 15, 15, 15, 15, 15, 90, 15, 15, 15, 15, 15, duration - 250])

      net = STDPNet(1, 1)
      def run(i, I_pre, I_post):
        pre_spike, post_spike, g, Apre, Apost, current, W = net.step_run(i, I_pre, I_post)
        return pre_spike, post_spike, g, Apre, Apost, current, W

      indices = bm.arange(0, duration, bm.dt)
      pre_spike, post_spike, g, Apre, Apost, current, W = bm.for_loop(run, [indices, I_pre, I_post])

    Args:
      tau_s: float. The time constant of :math:`A_{pre}`.
      tau_t: float. The time constant of :math:`A_{post}`.
      A1: float. The increment of :math:`A_{pre}` produced by a spike. Must be a positive value.
      A2: float. The increment of :math:`A_{post}` produced by a spike. Must be a positive value.
      W_max: float. The maximum weight.
      W_min: float. The minimum weight.
      pre: DynamicalSystem. The pre-synaptic neuron group.
      delay: int, float. The pre spike delay length. (ms)
      syn: DynamicalSystem. The synapse model.
      comm: DynamicalSystem. The communication model, for example, dense or sparse connection layers.
      out: DynamicalSystem. The synaptic current output models.
      post: DynamicalSystem. The post-synaptic neuron group.
      out_label: str. The output label.
      name: str. The model name.
    """

    def __init__(
        self,
        pre: JointType[DynamicalSystem, SupportAutoDelay],
        delay: Union[None, int, float],
        syn: ParamDescriber[DynamicalSystem],
        comm: JointType[DynamicalSystem, SupportSTDP],
        out: ParamDescriber[JointType[DynamicalSystem, BindCondData]],
        post: DynamicalSystem,
        # synapse parameters
        tau_s: Union[float, ArrayType, Callable] = 16.8,
        tau_t: Union[float, ArrayType, Callable] = 33.7,
        A1: Union[float, ArrayType, Callable] = 0.96,
        A2: Union[float, ArrayType, Callable] = 0.53,
        W_max: Optional[float] = None,
        W_min: Optional[float] = None,
        # others
        out_label: Optional[str] = None,
        name: Optional[str] = None,
        mode: Optional[bm.Mode] = None,
    ):
        super().__init__(name=name, mode=mode)

        # synaptic models
        check.is_instance(pre, JointType[DynamicalSystem, SupportAutoDelay])
        check.is_instance(comm, JointType[DynamicalSystem, SupportSTDP])
        check.is_instance(syn, ParamDescriber[DynamicalSystem])
        check.is_instance(out, ParamDescriber[JointType[DynamicalSystem, BindCondData]])
        check.is_instance(post, DynamicalSystem)
        self.pre_num = pre.num
        self.post_num = post.num
        self.comm = comm
        self._is_align_post = issubclass(syn.cls, AlignPost)

        # delay initialization
        delay_cls = register_delay_by_return(pre)
        delay_cls.register_entry(self.name, delay)

        # synapse and output initialization
        if self._is_align_post:
            syn_cls, out_cls = align_post_add_bef_update(out_label, syn_desc=syn, out_desc=out, post=post,
                                                         proj_name=self.name)
        else:
            syn_cls = align_pre2_add_bef_update(syn, delay, delay_cls, self.name + '-pre')
            out_cls = out()
            post.add_inp_fun(self.name, out_cls, label=out_label)

        # references
        self.refs = dict(pre=pre, post=post)  # invisible to ``self.nodes()``
        self.refs['delay'] = delay_cls
        self.refs['syn'] = syn_cls  # invisible to ``self.node()``
        self.refs['out'] = out_cls  # invisible to ``self.node()``
        self.refs['comm'] = comm

        # tracing pre-synaptic spikes using Exponential model
        self.refs['pre_trace'] = _init_trace_by_align_pre2(pre, delay, Expon.desc(pre.num, tau=tau_s))

        # tracing post-synaptic spikes using Exponential model
        self.refs['post_trace'] = _init_trace_by_align_pre2(post, None, Expon.desc(post.num, tau=tau_t))

        # synapse parameters
        self.W_max = W_max
        self.W_min = W_min
        self.tau_s = tau_s
        self.tau_t = tau_t
        self.A1 = A1
        self.A2 = A2

    pre = property(lambda self: self.refs['pre'])
    post = property(lambda self: self.refs['post'])
    syn = property(lambda self: self.refs['syn'])
    delay = property(lambda self: self.refs['delay'])
    out = property(lambda self: self.refs['out'])

    def update(self):
        # pre-synaptic spikes
        pre_spike = self.refs['delay'].at(self.name)  # spike
        # pre-synaptic variables
        if self._is_align_post:
            # For AlignPost, we need "pre spikes @ comm matrix" for computing post-synaptic conductance
            x = pre_spike
        else:
            # For AlignPre, we need the "pre synapse variable @ comm matrix" for computing post conductance
            x = _get_return(self.refs['syn'].return_info())  # pre-synaptic variable

        # post spikes
        if not hasattr(self.refs['post'], 'spike'):
            raise AttributeError(f'{self} needs a "spike" variable for the post-synaptic neuron group.')
        post_spike = self.refs['post'].spike.value

        # weight updates
        Apost = self.refs['post_trace'].g.value
        self.comm.stdp_update(
            on_pre={"spike": bm.as_jax(pre_spike), "trace": bm.as_jax(-Apost * self.A2)},
            w_min=bm.as_jax(self.W_min),
            w_max=bm.as_jax(self.W_max),
        )
        Apre = self.refs['pre_trace'].g.value
        self.comm.stdp_update(
            on_post={"spike": bm.as_jax(post_spike), "trace": bm.as_jax(Apre * self.A1)},
            w_min=bm.as_jax(self.W_min),
            w_max=bm.as_jax(self.W_max),
        )

        # synaptic currents
        current = self.comm(x)
        if self._is_align_post:
            self.refs['syn'].add_current(current)  # synapse post current
        else:
            self.refs['out'].bind_cond(current)  # align pre
        return current

# class PairedSTDP(Projection):
#   r"""Paired spike-time-dependent plasticity model.
#
#   This model filters the synaptic currents according to the variables: :math:`w`.
#
#   .. math::
#
#      I_{syn}^+(t) = I_{syn}^-(t) * w
#
#   where :math:`I_{syn}^-(t)` and :math:`I_{syn}^+(t)` are the synaptic currents before
#   and after STDP filtering, :math:`w` measures synaptic efficacy because each time a presynaptic neuron emits a pulse,
#   the conductance of the synapse will increase w.
#
#   The dynamics of :math:`w` is governed by the following equation:
#
#   .. math::
#
#       \begin{aligned}
#       \frac{dw}{dt} & = & -A_{post}\delta(t-t_{sp}) + A_{pre}\delta(t-t_{sp}), \\
#       \frac{dA_{pre}}{dt} & = & -\frac{A_{pre}}{\tau_s} + A_1\delta(t-t_{sp}), \\
#       \frac{dA_{post}}{dt} & = & -\frac{A_{post}}{\tau_t} + A_2\delta(t-t_{sp}), \\
#       \end{aligned}
#
#   where :math:`t_{sp}` denotes the spike time and :math:`A_1` is the increment
#   of :math:`A_{pre}`, :math:`A_2` is the increment of :math:`A_{post}` produced by a spike.
#
#   Here is an example of the usage of this class::
#
#     import brainpy as bp
#     import brainpy.math as bm
#
#     class STDPNet(bp.DynamicalSystem):
#        def __init__(self, num_pre, num_post):
#          super().__init__()
#          self.pre = bp.dyn.LifRef(num_pre)
#          self.post = bp.dyn.LifRef(num_post)
#          self.syn = bp.dyn.STDP_Song2000(
#            pre=self.pre,
#            delay=1.,
#            comm=bp.dnn.EventCSRLinear(bp.conn.FixedProb(1, pre=self.pre.num, post=self.post.num),
#                                       weight=bp.init.Uniform(max_val=0.1)),
#            syn=bp.dyn.Expon.desc(self.post.varshape, tau=5.),
#            out=bp.dyn.COBA.desc(E=0.),
#            post=self.post,
#            tau_s=16.8,
#            tau_t=33.7,
#            A1=0.96,
#            A2=0.53,
#          )
#
#        def update(self, I_pre, I_post):
#          self.syn()
#          self.pre(I_pre)
#          self.post(I_post)
#          conductance = self.syn.refs['syn'].g
#          Apre = self.syn.refs['pre_trace'].g
#          Apost = self.syn.refs['post_trace'].g
#          current = self.post.sum_inputs(self.post.V)
#          return self.pre.spike, self.post.spike, conductance, Apre, Apost, current, self.syn.comm.weight
#
#     duration = 300.
#     I_pre = bp.inputs.section_input([0, 30, 0, 30, 0, 30, 0, 30, 0, 30, 0, 30, 0],
#                                     [5, 15, 15, 15, 15, 15, 100, 15, 15, 15, 15, 15, duration - 255])
#     I_post = bp.inputs.section_input([0, 30, 0, 30, 0, 30, 0, 30, 0, 30, 0, 30, 0],
#                                      [10, 15, 15, 15, 15, 15, 90, 15, 15, 15, 15, 15, duration - 250])
#
#     net = STDPNet(1, 1)
#     def run(i, I_pre, I_post):
#       pre_spike, post_spike, g, Apre, Apost, current, W = net.step_run(i, I_pre, I_post)
#       return pre_spike, post_spike, g, Apre, Apost, current, W
#
#     indices = bm.arange(0, duration, bm.dt)
#     pre_spike, post_spike, g, Apre, Apost, current, W = bm.for_loop(run, [indices, I_pre, I_post])
#
#   Args:
#     tau_s: float. The time constant of :math:`A_{pre}`.
#     tau_t: float. The time constant of :math:`A_{post}`.
#     A1: float. The increment of :math:`A_{pre}` produced by a spike. Must be a positive value.
#     A2: float. The increment of :math:`A_{post}` produced by a spike. Must be a positive value.
#     W_max: float. The maximum weight.
#     W_min: float. The minimum weight.
#     pre: DynamicalSystem. The pre-synaptic neuron group.
#     delay: int, float. The pre spike delay length. (ms)
#     syn: DynamicalSystem. The synapse model.
#     comm: DynamicalSystem. The communication model, for example, dense or sparse connection layers.
#     out: DynamicalSystem. The synaptic current output models.
#     post: DynamicalSystem. The post-synaptic neuron group.
#     out_label: str. The output label.
#     name: str. The model name.
#   """
#
#   def __init__(
#       self,
#       pre: JointType[DynamicalSystem, SupportAutoDelay],
#       delay: Union[None, int, float],
#       syn: ParamDescriber[DynamicalSystem],
#       comm: JointType[DynamicalSystem, SupportSTDP],
#       out: ParamDescriber[JointType[DynamicalSystem, BindCondData]],
#       post: DynamicalSystem,
#       # synapse parameters
#       tau_s: float = 16.8,
#       tau_t: float = 33.7,
#       lambda_: float = 0.96,
#       alpha: float = 0.53,
#       mu: float = 0.53,
#       W_max: Optional[float] = None,
#       W_min: Optional[float] = None,
#       # others
#       out_label: Optional[str] = None,
#       name: Optional[str] = None,
#       mode: Optional[bm.Mode] = None,
#   ):
#     super().__init__(name=name, mode=mode)
#
#     # synaptic models
#     check.is_instance(pre, JointType[DynamicalSystem, SupportAutoDelay])
#     check.is_instance(comm, JointType[DynamicalSystem, SupportSTDP])
#     check.is_instance(syn, ParamDescriber[DynamicalSystem])
#     check.is_instance(out, ParamDescriber[JointType[DynamicalSystem, BindCondData]])
#     check.is_instance(post, DynamicalSystem)
#     self.pre_num = pre.num
#     self.post_num = post.num
#     self.comm = comm
#     self._is_align_post = issubclass(syn.cls, AlignPost)
#
#     # delay initialization
#     delay_cls = register_delay_by_return(pre)
#     delay_cls.register_entry(self.name, delay)
#
#     # synapse and output initialization
#     if self._is_align_post:
#       syn_cls, out_cls = align_post_add_bef_update(out_label, syn_desc=syn, out_desc=out, post=post,
#                                                    proj_name=self.name)
#     else:
#       syn_cls = align_pre2_add_bef_update(syn, delay, delay_cls, self.name + '-pre')
#       out_cls = out()
#       add_inp_fun(out_label, self.name, out_cls, post)
#
#     # references
#     self.refs = dict(pre=pre, post=post)  # invisible to ``self.nodes()``
#     self.refs['delay'] = delay_cls
#     self.refs['syn'] = syn_cls  # invisible to ``self.node()``
#     self.refs['out'] = out_cls  # invisible to ``self.node()``
#     self.refs['comm'] = comm
#
#     # tracing pre-synaptic spikes using Exponential model
#     self.refs['pre_trace'] = _init_trace_by_align_pre2(pre, delay, Expon.desc(pre.num, tau=tau_s))
#
#     # tracing post-synaptic spikes using Exponential model
#     self.refs['post_trace'] = _init_trace_by_align_pre2(post, None, Expon.desc(post.num, tau=tau_t))
#
#     # synapse parameters
#     self.W_max = W_max
#     self.W_min = W_min
#     self.tau_s = tau_s
#     self.tau_t = tau_t
#     self.A1 = A1
#     self.A2 = A2
#
#   def update(self):
#     # pre-synaptic spikes
#     pre_spike = self.refs['delay'].at(self.name)  # spike
#     # pre-synaptic variables
#     if self._is_align_post:
#       # For AlignPost, we need "pre spikes @ comm matrix" for computing post-synaptic conductance
#       x = pre_spike
#     else:
#       # For AlignPre, we need the "pre synapse variable @ comm matrix" for computing post conductance
#       x = _get_return(self.refs['syn'].return_info())  # pre-synaptic variable
#
#     # post spikes
#     if not hasattr(self.refs['post'], 'spike'):
#       raise AttributeError(f'{self} needs a "spike" variable for the post-synaptic neuron group.')
#     post_spike = self.refs['post'].spike
#
#     # weight updates
#     Apost = self.refs['post_trace'].g
#     self.comm.stdp_update(on_pre={"spike": pre_spike, "trace": -Apost * self.A2}, w_min=self.W_min, w_max=self.W_max)
#     Apre = self.refs['pre_trace'].g
#     self.comm.stdp_update(on_post={"spike": post_spike, "trace": Apre * self.A1}, w_min=self.W_min, w_max=self.W_max)
#
#     # synaptic currents
#     current = self.comm(x)
#     if self._is_align_post:
#       self.refs['syn'].add_current(current)  # synapse post current
#     else:
#       self.refs['out'].bind_cond(current)  # align pre
#     return current
