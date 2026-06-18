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
"""Line-coverage tests for ``brainpy/dyn/others/common.py``.

Targets the ``Leaky`` and ``Integrator`` neuron-dynamics helpers. Exercises:

* construction in non-batching and training modes,
* ``init_var=False`` followed by manual ``reset_state(batch_size=...)``,
* ``update`` with an explicit input and with the default ``None`` input
  (the ``inp is None`` / ``x is None`` branches),
* ``derivative`` and ``return_info``,
* a short ``DSRunner`` simulation to drive the integrators end-to-end.
"""

from absl.testing import parameterized

import brainpy as bp
import brainpy.math as bm
from brainpy.context import share
from brainpy.dyn.others.common import Leaky, Integrator


class TestLeaky(parameterized.TestCase):
    def test_nonbatching_runner(self):
        bm.random.seed()
        model = Leaky(4, tau=8.)
        self.assertTupleEqual(tuple(model.x.shape), (4,))
        runner = bp.DSRunner(model, monitors=['x'], progress_bar=False)
        runner.run(5.)
        self.assertTupleEqual(runner.mon['x'].shape, (50, 4))

    def test_derivative_and_return_info(self):
        model = Leaky(3, tau=10.)
        dx = model.derivative(model.x.value, 0.)
        self.assertTupleEqual(tuple(dx.shape), (3,))
        # return_info hands back the state variable itself
        self.assertIs(model.return_info(), model.x)

    def test_training_update_with_and_without_input(self):
        # init_var=False exercises the "do not reset in __init__" branch,
        # then a manual batched reset_state.
        model = Leaky(3, mode=bm.TrainingMode(), tau=5.,
                      method='exp_euler', init_var=False)
        model.reset_state(batch_size=2)
        self.assertTupleEqual(tuple(model.x.shape), (2, 3))

        share.save(t=0., dt=0.1)
        out = model.update(bm.ones((2, 3)))   # inp is not None branch
        self.assertTupleEqual(tuple(out.shape), (2, 3))
        out_none = model.update()              # inp is None branch
        self.assertTupleEqual(tuple(out_none.shape), (2, 3))


class TestIntegrator(parameterized.TestCase):
    def test_nonbatching_runner(self):
        bm.random.seed()
        model = Integrator(4, tau=8.)
        self.assertTupleEqual(tuple(model.x.shape), (4,))
        runner = bp.DSRunner(model, monitors=['x'], progress_bar=False)
        runner.run(5.)
        self.assertTupleEqual(runner.mon['x'].shape, (50, 4))

    def test_derivative_and_return_info(self):
        model = Integrator(3, tau=10.)
        dv = model.derivative(model.x.value, 0., 1.0)
        self.assertTupleEqual(tuple(dv.shape), (3,))
        self.assertIs(model.return_info(), model.x)

    def test_custom_initializer(self):
        model = Integrator(3, x_initializer=bp.init.Constant(0.5))
        self.assertTrue(bm.allclose(bm.as_jax(model.x.value),
                                    bm.as_jax(bm.ones(3) * 0.5)))

    def test_training_update_with_and_without_input(self):
        model = Integrator(3, mode=bm.TrainingMode(), tau=5., init_var=False)
        model.reset_state(batch_size=2)
        self.assertTupleEqual(tuple(model.x.shape), (2, 3))

        share.save(t=0., dt=0.1)
        out = model.update(bm.ones((2, 3)))   # x is not None branch
        self.assertTupleEqual(tuple(out.shape), (2, 3))
        out_none = model.update()              # x is None branch
        self.assertTupleEqual(tuple(out_none.shape), (2, 3))


if __name__ == '__main__':
    parameterized.absltest.main()
