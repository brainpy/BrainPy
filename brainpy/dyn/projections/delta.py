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
from typing import Optional, Union

from brainpy import math as bm, check
from brainpy.delay import (delay_identifier, register_delay_by_return)
from brainpy.dynsys import DynamicalSystem, Projection
from brainpy.mixin import (JointType, SupportAutoDelay)

__all__ = [
    'HalfProjDelta', 'FullProjDelta',
]


class _Delta:
    def __init__(self):
        self._cond = None

    def bind_cond(self, cond):
        self._cond = cond

    def __call__(self, *args, **kwargs):
        r = self._cond
        return r


class HalfProjDelta(Projection):
    r"""Defining the half-part of the synaptic projection for the Delta synapse model.

    The synaptic projection requires the input is the spiking data, otherwise
    the synapse is not the Delta synapse model.

    The ``half-part`` means that the model only includes ``comm`` -> ``syn`` -> ``out`` -> ``post``.
    Therefore, the model's ``update`` function needs the manual providing of the spiking input.

    **Model Descriptions**

    .. math::

        I_{syn} (t) = \sum_{j\in C} g_{\mathrm{max}} * \delta(t-t_j-D)

    where :math:`g_{\mathrm{max}}` denotes the chemical synaptic strength,
    :math:`t_j` the spiking moment of the presynaptic neuron :math:`j`,
    :math:`C` the set of neurons connected to the post-synaptic neuron,
    and :math:`D` the transmission delay of chemical synapses.
    For simplicity, the rise and decay phases of post-synaptic currents are
    omitted in this model.


    **Code Examples**

    .. code-block::

        import brainpy as bp
        import brainpy.math as bm

        class Net(bp.DynamicalSystem):
          def __init__(self):
            super().__init__()

            self.pre = bp.dyn.PoissonGroup(10, 100.)
            self.post = bp.dyn.LifRef(1)
            self.syn = bp.dyn.HalfProjDelta(bp.dnn.Linear(10, 1, bp.init.OneInit(2.)), self.post)

          def update(self):
            self.syn(self.pre())
            self.post()
            return self.post.V.value

        net = Net()
        indices = bm.arange(1000).to_numpy()
        vs = bm.for_loop(net.step_run, indices, progress_bar=True)
        bp.visualize.line_plot(indices, vs, show=True)

    Args:
      comm: DynamicalSystem. The synaptic communication.
      post: DynamicalSystem. The post-synaptic neuron group.
      name: str. The projection name.
      mode: Mode. The computing mode.
    """

    def __init__(
        self,
        comm: DynamicalSystem,
        post: DynamicalSystem,
        name: Optional[str] = None,
        mode: Optional[bm.Mode] = None,
    ):
        super().__init__(name=name, mode=mode)

        # synaptic models
        check.is_instance(comm, DynamicalSystem)
        check.is_instance(post, DynamicalSystem)
        self.comm = comm

        # output initialization
        out = _Delta()
        post.add_inp_fun(self.name, out, category='delta')

        # references
        self.refs = dict(post=post, out=out)  # invisible to ``self.nodes()``
        self.refs['comm'] = comm  # unify the access

    def update(self, x):
        # call the communication
        current = self.comm(x)
        # bind the output
        self.refs['out'].bind_cond(current)
        # return the current, if needed
        return current


class FullProjDelta(Projection):
    r"""Full-chain of the synaptic projection for the Delta synapse model.

    The synaptic projection requires the input is the spiking data, otherwise
    the synapse is not the Delta synapse model.

    The ``full-chain`` means that the model needs to provide all information needed for a projection,
    including ``pre`` -> ``delay`` -> ``comm`` -> ``post``.

    **Model Descriptions**

    .. math::

        I_{syn} (t) = \sum_{j\in C} g_{\mathrm{max}} * \delta(t-t_j-D)

    where :math:`g_{\mathrm{max}}` denotes the chemical synaptic strength,
    :math:`t_j` the spiking moment of the presynaptic neuron :math:`j`,
    :math:`C` the set of neurons connected to the post-synaptic neuron,
    and :math:`D` the transmission delay of chemical synapses.
    For simplicity, the rise and decay phases of post-synaptic currents are
    omitted in this model.


    **Code Examples**

    .. code-block::

        import brainpy as bp
        import brainpy.math as bm


        class Net(bp.DynamicalSystem):
          def __init__(self):
            super().__init__()

            self.pre = bp.dyn.PoissonGroup(10, 100.)
            self.post = bp.dyn.LifRef(1)
            self.syn = bp.dyn.FullProjDelta(self.pre, 0., bp.dnn.Linear(10, 1, bp.init.OneInit(2.)), self.post)

          def update(self):
            self.syn()
            self.pre()
            self.post()
            return self.post.V.value


        net = Net()
        indices = bm.arange(1000).to_numpy()
        vs = bm.for_loop(net.step_run, indices, progress_bar=True)
        bp.visualize.line_plot(indices, vs, show=True)


    Args:
      pre: The pre-synaptic neuron group.
      delay: The synaptic delay.
      comm: DynamicalSystem. The synaptic communication.
      post: DynamicalSystem. The post-synaptic neuron group.
      name: str. The projection name.
      mode: Mode. The computing mode.
    """

    def __init__(
        self,
        pre: JointType[DynamicalSystem, SupportAutoDelay],
        delay: Union[None, int, float],
        comm: DynamicalSystem,
        post: DynamicalSystem,
        name: Optional[str] = None,
        mode: Optional[bm.Mode] = None,
    ):
        super().__init__(name=name, mode=mode)

        # synaptic models
        check.is_instance(pre, JointType[DynamicalSystem, SupportAutoDelay])
        check.is_instance(comm, DynamicalSystem)
        check.is_instance(post, DynamicalSystem)
        self.comm = comm

        # delay initialization
        delay_cls = register_delay_by_return(pre)
        delay_cls.register_entry(self.name, delay)

        # output initialization
        out = _Delta()
        post.add_inp_fun(self.name, out, category='delta')

        # references
        self.refs = dict(pre=pre, post=post, out=out)  # invisible to ``self.nodes()``
        self.refs['comm'] = comm  # unify the access
        self.refs['delay'] = pre.get_aft_update(delay_identifier)

    def update(self):
        # get delay
        x = self.refs['pre'].get_aft_update(delay_identifier).at(self.name)
        # call the communication
        current = self.comm(x)
        # bind the output
        self.refs['out'].bind_cond(current)
        # return the current, if needed
        return current
