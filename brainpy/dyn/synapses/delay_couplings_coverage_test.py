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
"""Line-coverage tests for ``brainpy/dyn/synapses/delay_couplings.py``.

Covers ``DelayCoupling`` (base), ``DiffusiveCoupling`` and ``AdditiveCoupling``.
The existing ``delay_couplings_test.py`` only drives the ``delay_steps=None``
path, so here we add:

* the ``delay_type == 'int'`` path (scalar integer ``delay_steps``),
* the ``delay_type == 'array'`` path (integer delay matrix),
* a *callable* ``delay_steps`` that returns an integer delay matrix,
* ``DelayCoupling.reset_state`` (the no-op),
* the single-``Variable`` ``var_to_output`` wrapping branch,
* and every constructor error branch:
    - ``delay_var`` / ``coupling_var*`` not a ``brainpy.math.Variable``,
    - coupling variable not 1-D,
    - connection-matrix shape mismatch,
    - non-integer ``delay_steps`` array (and callable returning non-int),
    - ``delay_steps`` array shape mismatch,
    - unknown ``delay_steps`` type.

NOTE (defect, not fixed here -- source must not be modified):
    The ``TrainingMode`` + ``delay_type == 'array'`` branches of
    ``DiffusiveCoupling.update`` (delay_couplings.py:212-213) and
    ``AdditiveCoupling.update`` (delay_couplings.py:295-296) raise at runtime.
    Under that branch the ``vmap``-ed ``delay_var.retrieve(steps, *indices)``
    call receives ``delay_step=None`` (the lambda passes ``steps`` positionally
    but ``Delay.retrieve`` then computes ``delay_step - i - 1`` with
    ``delay_step is None``), giving
    ``TypeError: unsupported operand type(s) for -: 'BatchTracer' and 'NoneType'``.
    These two lines are therefore left uncovered.
"""

import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

import brainpy as bp
import brainpy.math as bm
from brainpy.dyn.synapses.delay_couplings import DelayCoupling


N = 5


def _make_area(name):
    bm.set_dt(0.1)
    return bp.rates.FHN(N, x_ou_sigma=0.01, y_ou_sigma=0.01, name=name)


def _conn_mat():
    return bp.conn.All2All(pre=N, post=N).require('conn_mat')


def _int_delay_matrix(seed=0):
    rng = np.random.RandomState(seed)
    return bm.asarray(rng.randint(0, 4, size=(N, N)), dtype=bm.int_)


class TestDiffusiveCoupling(parameterized.TestCase):
    def _run(self, conn, area, name):
        net = bp.Network(area, conn)
        runner = bp.DSRunner(net, monitors=[f'{name}.x'],
                             inputs=(f'{name}.input', 5.), progress_bar=False)
        runner(2.)
        self.assertTupleEqual(runner.mon[f'{name}.x'].shape, (20, N))

    def test_int_delay(self):
        bm.random.seed()
        area = _make_area('diffA')
        conn = bp.synapses.DiffusiveCoupling(
            area.x, area.x, area.input, conn_mat=_conn_mat(),
            delay_steps=3, initial_delay_data=bp.init.Uniform(0, 0.05))
        self.assertEqual(conn.delay_type, 'int')
        self._run(conn, area, 'diffA')

    def test_array_delay(self):
        bm.random.seed()
        area = _make_area('diffB')
        conn = bp.synapses.DiffusiveCoupling(
            area.x, area.x, area.input, conn_mat=_conn_mat(),
            delay_steps=_int_delay_matrix(1),
            initial_delay_data=bp.init.Uniform(0, 0.05))
        self.assertEqual(conn.delay_type, 'array')
        self._run(conn, area, 'diffB')

    def test_callable_delay(self):
        bm.random.seed()
        area = _make_area('diffC')

        def make_delays(shape):
            rng = np.random.RandomState(2)
            return bm.asarray(rng.randint(0, 3, size=shape), dtype=bm.int_)

        conn = bp.synapses.DiffusiveCoupling(
            area.x, area.x, area.input, conn_mat=_conn_mat(),
            delay_steps=make_delays,
            initial_delay_data=bp.init.Uniform(0, 0.05))
        self.assertEqual(conn.delay_type, 'array')
        self._run(conn, area, 'diffC')

    def test_scalar_int_array_delay(self):
        # A 0-d integer *array* (ndim == 0) is classified as ``delay_type='int'``.
        bm.random.seed()
        area = _make_area('diffD')
        d0 = bm.asarray(np.int32(2))
        self.assertEqual(d0.ndim, 0)
        conn = bp.synapses.DiffusiveCoupling(
            area.x, area.x, area.input, conn_mat=_conn_mat(),
            delay_steps=d0, initial_delay_data=bp.init.Uniform(0, 0.05))
        self.assertEqual(conn.delay_type, 'int')
        self._run(conn, area, 'diffD')

    def test_diffusive_cv2_not_1d(self):
        # cv1 is 1-D, cv2 is 2-D -> the second ndim check fires.
        area = _make_area('diffE')
        v2d = bm.Variable(jnp.ones((2, N)))
        with self.assertRaises(ValueError):
            bp.synapses.DiffusiveCoupling(area.x, v2d, area.input,
                                          conn_mat=_conn_mat())


class TestAdditiveCoupling(parameterized.TestCase):
    def _run(self, conn, area, name):
        net = bp.Network(area, conn)
        runner = bp.DSRunner(net, monitors=[f'{name}.x'],
                             inputs=(f'{name}.input', 5.), progress_bar=False)
        runner(2.)
        self.assertTupleEqual(runner.mon[f'{name}.x'].shape, (20, N))

    def test_int_delay(self):
        bm.random.seed()
        area = _make_area('addA')
        conn = bp.synapses.AdditiveCoupling(
            area.x, area.input, conn_mat=_conn_mat(),
            delay_steps=2, initial_delay_data=bp.init.Uniform(0, 0.05))
        self.assertEqual(conn.delay_type, 'int')
        self._run(conn, area, 'addA')

    def test_array_delay(self):
        bm.random.seed()
        area = _make_area('addB')
        conn = bp.synapses.AdditiveCoupling(
            area.x, area.input, conn_mat=_conn_mat(),
            delay_steps=_int_delay_matrix(3),
            initial_delay_data=bp.init.Uniform(0, 0.05))
        self.assertEqual(conn.delay_type, 'array')
        self._run(conn, area, 'addB')


class TestDelayCouplingBase(parameterized.TestCase):
    def test_single_variable_output_wrapped_and_reset(self):
        area = _make_area('baseA')
        dc = DelayCoupling(area.x, area.input, conn_mat=_conn_mat(),
                           required_shape=(N, N))
        # a single Variable target gets wrapped into a list
        self.assertIsInstance(dc.output_var, list)
        self.assertEqual(len(dc.output_var), 1)
        # reset_state is a documented no-op; just exercise it
        self.assertIsNone(dc.reset_state())

    def test_base_delay_var_not_variable(self):
        area = _make_area('baseB')
        with self.assertRaises(ValueError):
            DelayCoupling(jnp.ones(N), area.input, conn_mat=_conn_mat(),
                          required_shape=(N, N))

    def test_callable_delay_returns_non_integer(self):
        area = _make_area('baseC')

        def bad_delays(shape):
            return bm.asarray(np.ones(shape), dtype=float)

        with self.assertRaises(ValueError):
            DelayCoupling(area.x, area.input, conn_mat=_conn_mat(),
                          required_shape=(N, N), delay_steps=bad_delays)

    def test_unknown_delay_steps_type(self):
        area = _make_area('baseD')
        with self.assertRaises(ValueError):
            DelayCoupling(area.x, area.input, conn_mat=_conn_mat(),
                          required_shape=(N, N), delay_steps='oops')


class TestCouplingErrors(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.area = _make_area(f'errArea_{self.id().rsplit(".", 1)[-1]}')
        self.cmat = _conn_mat()

    def test_diffusive_cv1_not_variable(self):
        with self.assertRaises(ValueError):
            bp.synapses.DiffusiveCoupling(jnp.ones(N), self.area.x,
                                          self.area.input, conn_mat=self.cmat)

    def test_diffusive_cv2_not_variable(self):
        with self.assertRaises(ValueError):
            bp.synapses.DiffusiveCoupling(self.area.x, jnp.ones(N),
                                          self.area.input, conn_mat=self.cmat)

    def test_diffusive_coupling_var_not_1d(self):
        v2d = bm.Variable(jnp.ones((2, N)))
        with self.assertRaises(ValueError):
            bp.synapses.DiffusiveCoupling(v2d, v2d, self.area.input,
                                          conn_mat=self.cmat)

    def test_conn_mat_shape_mismatch(self):
        with self.assertRaises(ValueError):
            bp.synapses.DiffusiveCoupling(self.area.x, self.area.x,
                                          self.area.input,
                                          conn_mat=bm.ones((N + 1, N)))

    def test_delay_steps_array_non_integer(self):
        with self.assertRaises(ValueError):
            bp.synapses.DiffusiveCoupling(self.area.x, self.area.x,
                                          self.area.input, conn_mat=self.cmat,
                                          delay_steps=bm.ones((N, N)))

    def test_delay_steps_array_shape_mismatch(self):
        bad = bm.asarray(np.ones((N + 1, N)), dtype=bm.int_)
        with self.assertRaises(ValueError):
            bp.synapses.DiffusiveCoupling(self.area.x, self.area.x,
                                          self.area.input, conn_mat=self.cmat,
                                          delay_steps=bad)

    def test_unknown_delay_steps_type(self):
        with self.assertRaises(ValueError):
            bp.synapses.DiffusiveCoupling(self.area.x, self.area.x,
                                          self.area.input, conn_mat=self.cmat,
                                          delay_steps='oops')

    def test_additive_coupling_var_not_variable(self):
        with self.assertRaises(ValueError):
            bp.synapses.AdditiveCoupling(jnp.ones(N), self.area.input,
                                         conn_mat=self.cmat)

    def test_additive_coupling_var_not_1d(self):
        v2d = bm.Variable(jnp.ones((2, N)))
        with self.assertRaises(ValueError):
            bp.synapses.AdditiveCoupling(v2d, self.area.input,
                                         conn_mat=self.cmat)


if __name__ == '__main__':
    parameterized.absltest.main()
