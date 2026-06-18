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
"""Branch-coverage tests for ``brainpy/dyn/neurons/lif.py``.

The existing ``lif_test.py`` drives every neuron in the default mode and in the
training mode with the *soft* reset.  This file targets the remaining
uncovered branches that are shared across all LIF families:

* the training-mode ``spk_reset='hard'`` branch,
* the ``else: raise ValueError`` branch for an unknown ``spk_reset`` mode,
* the ``ref_var=True`` branch of the refractory (``*Ref*``) variants, in both
  the default and the training modes (populates the ``refractory`` variable),
* the stochastic (``noise=`` -> ``sdeint``) integration path.

Classes are discovered from ``lif.__all__`` and bucketed by capability so the
tests stay correct as the module evolves.
"""

import numpy as np
from absl.testing import parameterized

import brainpy as bp
import brainpy.math as bm
from brainpy.dyn.neurons import lif

# IF / IFLTC are pure integrators (no threshold reset), and the Izhikevich
# family resets ``V``/``u`` directly without a soft/hard ``spk_reset`` switch,
# so neither participates in the hard-reset / invalid-reset branches.
_NO_RESET_SWITCH = ('IF', 'IFLTC', 'Izhikevich', 'IzhikevichLTC',
                    'IzhikevichRef', 'IzhikevichRefLTC')

_RESET_SWITCH_CLASSES = [n for n in lif.__all__ if n not in _NO_RESET_SWITCH]
_REF_CLASSES = [n for n in lif.__all__ if 'Ref' in n]
# Classes accepting a ``noise=`` keyword (everything except IF / IFLTC).
_NOISE_CLASSES = [n for n in lif.__all__ if n not in ('IF', 'IFLTC')]


def _run(model, steps=30, current=25.):
    return bm.for_loop(lambda i: model.step_run(i, current), np.arange(steps))


class TestSpkResetBranches(parameterized.TestCase):
    @parameterized.named_parameters(
        {'testcase_name': name, 'neuron': name} for name in _RESET_SWITCH_CLASSES
    )
    def test_training_hard_reset(self, neuron):
        bm.random.seed()
        bm.set_dt(0.1)
        model = getattr(lif, neuron)(size=3, mode=bm.training_mode,
                                     spk_reset='hard')
        out = _run(model)
        self.assertEqual(out.shape[0], 30)

    @parameterized.named_parameters(
        {'testcase_name': name, 'neuron': name} for name in _RESET_SWITCH_CLASSES
    )
    def test_invalid_spk_reset_raises(self, neuron):
        bm.random.seed()
        bm.set_dt(0.1)
        model = getattr(lif, neuron)(size=3, mode=bm.training_mode,
                                     spk_reset='not-a-mode')
        with self.assertRaises(ValueError):
            _run(model, steps=3)


class TestRefVarBranches(parameterized.TestCase):
    @parameterized.named_parameters(
        {'testcase_name': name, 'neuron': name} for name in _REF_CLASSES
    )
    def test_ref_var_default_mode(self, neuron):
        bm.random.seed()
        bm.set_dt(0.1)
        model = getattr(lif, neuron)(size=3, tau_ref=2., ref_var=True)
        self.assertTrue(hasattr(model, 'refractory'))
        out = _run(model, steps=40)
        self.assertEqual(out.shape[0], 40)

    @parameterized.named_parameters(
        {'testcase_name': name, 'neuron': name} for name in _REF_CLASSES
    )
    def test_ref_var_training_mode(self, neuron):
        bm.random.seed()
        bm.set_dt(0.1)
        model = getattr(lif, neuron)(size=3, tau_ref=2., ref_var=True,
                                     mode=bm.training_mode)
        self.assertTrue(hasattr(model, 'refractory'))
        out = _run(model, steps=40)
        self.assertEqual(out.shape[0], 40)


class TestNoiseBranch(parameterized.TestCase):
    @parameterized.named_parameters(
        {'testcase_name': name, 'neuron': name} for name in _NOISE_CLASSES
    )
    def test_noise_sdeint(self, neuron):
        bm.random.seed()
        bm.set_dt(0.1)
        model = getattr(lif, neuron)(size=3, noise=0.5)
        # the sde integral should have been built
        out = _run(model, steps=20)
        self.assertEqual(out.shape[0], 20)


class TestLTCReturnInfo(parameterized.TestCase):
    # ``return_info`` lives on the LTC base classes; the running tests only
    # instantiate the non-LTC subclasses (which inherit it), so the base-class
    # source lines stay uncovered unless the LTC base is instantiated directly.
    @parameterized.named_parameters(
        {'testcase_name': name, 'neuron': name}
        for name in ('ExpIFLTC', 'AdExIFLTC', 'QuaIFLTC', 'AdQuaIFLTC',
                     'GifLTC', 'IzhikevichLTC')
    )
    def test_ltc_return_info_is_spike(self, neuron):
        bm.set_dt(0.1)
        model = getattr(lif, neuron)(size=2)
        self.assertIs(model.return_info(), model.spike)


class TestIFDerivativeOverride(parameterized.TestCase):
    def test_if_runs(self):
        # IF overrides ``derivative`` / ``update`` (no LTC current injection).
        bm.random.seed()
        bm.set_dt(0.1)
        model = lif.IF(size=3)
        out = _run(model, steps=20, current=5.)
        self.assertEqual(out.shape[0], 20)
        self.assertIs(model.return_info(), model.V)


if __name__ == '__main__':
    parameterized.absltest.main()
