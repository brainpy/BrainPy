# -*- coding: utf-8 -*-
"""Audit regression + coverage tests (audit ``docs/issues-found-20260618.md``).

This module exercises the fixes recorded in the 2026-06-18 BrainPy audit for the
rate-population / reservoir / RNN-cell modules under ``brainpy/dyn/rates`` and the
``brainpy/dynold`` compatibility shims.

Regression behaviors covered:

* C-15 ``ThresholdLinearModel`` noise path no longer crashes (``randn(*shape)``).
* C-16 ``StuartLandauOscillator.dy`` uses the correct ``+w*x`` rotational coupling.
* C-17 (dynold copy in ``experimental/others.py``) ``PoissonInput`` Gaussian branch
  uses ``std = sqrt(b*p)``, not the variance ``b*p``.
* C-20 ``AlphaCUBA`` / ``AlphaCOBA`` construct without ``ZeroDivisionError``.
* C-21 dynold ``STP`` synaptic current does not drift with zero presynaptic spikes.
* H-36 ``LSTMCell`` ``h`` / ``c`` setters slice the last axis (unbatched + batched);
  setting ``c`` does not corrupt ``h``.
* H-37 ``Reservoir`` recurrent noise is symmetric (``uniform(-1, 1)`` -> zero mean).
* H-38 ``Reservoir`` bias is actually added in ``update``.

The remaining tests construct + step the assigned modules for coverage. Bugs that
are still *unfixed* in the source are recorded in the agent summary, not asserted
here (e.g. M-21 ``reset_state(None)``, M-24 ``ALIFBellec2020`` ``a_initializer``).
"""

import jax.numpy as jnp
import pytest

import brainpy as bp
import brainpy.math as bm
import brainpy.initialize as init
from brainpy.context import share


def _share(t=0.0, dt=0.1, i=0):
    """Populate the shared simulation context required by ``update``."""
    bm.set_dt(dt)
    share.save(t=t, dt=dt, i=i)


# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------

def test_threshold_linear_model_noise_path_runs():
    """C-15: nonzero ``noise_e`` no longer raises ``TypeError`` from ``randn``."""
    bm.random.seed(0)
    m = bp.dyn.ThresholdLinearModel(5, noise_e=1.0)
    _share(t=0.0, dt=1e-4, i=0)
    out = m.update(inp_e=0.0)
    arr = jnp.asarray(out)
    assert arr.shape == (5,)
    assert bool(jnp.isfinite(arr).all())


def test_threshold_linear_model_noise_i_path_runs():
    """C-15 companion: the inhibitory noise branch is also reachable."""
    bm.random.seed(1)
    m = bp.dyn.ThresholdLinearModel(4, noise_i=0.5)
    _share(t=0.0, dt=1e-4, i=0)
    out = m.update(inp_e=1.0, inp_i=1.0)
    assert bool(jnp.isfinite(jnp.asarray(out)).all())


def test_stuart_landau_dy_rotational_coupling():
    """C-16: ``dy`` must add ``+w*x`` (Hopf normal form), not ``-w*y``."""
    m = bp.dyn.StuartLandauOscillator(1, a=0.25, w=0.2)
    # signature is dy(self, y, t, x, y_ext, a, w)
    val = float(jnp.asarray(m.dy(0.3, 0.0, 0.5, 0.0, 0.25, 0.2)))
    # (a - x^2 - y^2)*y + w*x = (0.25 - 0.25 - 0.09)*0.3 + 0.2*0.5 = 0.073
    assert val == pytest.approx(0.073, abs=1e-4)
    # The buggy -w*y value would be -0.087; make sure we are not there.
    assert val > 0.0


def test_poisson_input_gaussian_std_is_sqrt_bp():
    """C-17 (dynold copy): Gaussian branch std = sqrt(b*p), ~4.43 not ~19.6."""
    from brainpy.dynold.experimental.others import PoissonInput

    bm.random.seed(0)
    dt = 0.1
    freq, num = 200.0, 1000
    pi = PoissonInput(target_shape=(20000,), num_input=num, freq=freq, weight=1.0)
    _share(t=0.0, dt=dt, i=0)
    out = jnp.asarray(pi.update())

    p = freq * dt / 1e3            # 0.02
    b = num * (1 - p)
    expected_std = float((b * p) ** 0.5)   # ~4.427
    expected_mean = num * p                # 20.0

    assert expected_std == pytest.approx(4.427, abs=1e-2)
    # empirical std close to sqrt(b*p), and clearly far from the variance b*p (=19.6)
    assert float(out.std()) == pytest.approx(expected_std, rel=0.1)
    assert float(out.mean()) == pytest.approx(expected_mean, rel=0.1)


def test_alpha_cuba_coba_construct_and_step():
    """C-20: AlphaCUBA/COBA construct without ZeroDivisionError and run a step."""
    bm.random.seed(0)
    bm.set_dt(0.1)

    pre = bp.neurons.LIF(2)
    post = bp.neurons.LIF(2)
    syn = bp.synapses.AlphaCUBA(pre, post, bp.connect.All2All(), tau_decay=10.0)
    net = bp.Network(pre=pre, post=post, syn=syn)
    net.reset_state()
    _share()
    syn.update()  # must not raise

    pre2 = bp.neurons.LIF(2)
    post2 = bp.neurons.LIF(2)
    syn2 = bp.synapses.AlphaCOBA(pre2, post2, bp.connect.All2All(), tau_decay=10.0)
    net2 = bp.Network(pre=pre2, post=post2, syn=syn2)
    net2.reset_state()
    _share()
    syn2.update()  # must not raise


def test_dynold_stp_no_drift_without_spikes():
    """C-21: dynold STP synaptic current stays at 0 with zero presynaptic spikes."""
    bm.random.seed(0)
    bm.set_dt(0.1)
    pre = bp.neurons.LIF(1)
    post = bp.neurons.LIF(1)
    syn = bp.synapses.STP(pre, post, bp.connect.All2All(), U=0.2, tau_d=150.0, tau_f=2.0)
    net = bp.Network(pre=pre, syn=syn, post=post)
    net.reset_state()

    currents = []
    for i in range(100):
        _share(t=i * 0.1, dt=0.1, i=i)
        net.update()  # no external input -> no spikes
        currents.append(float(jnp.asarray(syn.I.value).sum()))

    # With the spike-gating fix, the current must not ramp up.
    assert max(abs(c) for c in currents) < 1e-5


def test_dynold_stp_jumps_with_spikes():
    """C-21 companion: STP current does jump when presynaptic spikes occur."""
    bm.random.seed(0)
    bm.set_dt(0.1)
    pre = bp.neurons.LIF(1)
    post = bp.neurons.LIF(1)
    syn = bp.synapses.STP(pre, post, bp.connect.All2All(), U=0.2, tau_d=150.0, tau_f=2.0)
    net = bp.Network(pre=pre, syn=syn, post=post)
    runner = bp.DSRunner(net, inputs=[('pre.input', 28.0)], monitors=['syn.I'],
                         progress_bar=False)
    runner.run(50.0)
    I = runner.mon['syn.I']
    assert bool(jnp.isfinite(I).all())
    assert float(I.max()) > 0.0


def test_lstm_h_c_setters_unbatched():
    """H-36: unbatched ``h``/``c`` setters slice the last axis (no IndexError)."""
    bm.random.seed(0)
    lstm = bp.dyn.LSTMCell(3, 4)
    assert lstm.state.shape == (8,)

    lstm.h = jnp.ones((4,))
    assert jnp.allclose(jnp.asarray(lstm.h), 1.0)
    # setting h must not have touched c
    assert jnp.allclose(jnp.asarray(lstm.c), 0.0)

    lstm.c = jnp.full((4,), 2.0)
    # setting c must not corrupt h
    assert jnp.allclose(jnp.asarray(lstm.h), 1.0)
    assert jnp.allclose(jnp.asarray(lstm.c), 2.0)


def test_lstm_h_c_setters_batched():
    """H-36: batched ``h``/``c`` setters write the correct rows."""
    bm.random.seed(0)
    lstm = bp.dyn.LSTMCell(3, 4, mode=bm.batching_mode)
    lstm.reset_state(2)
    assert lstm.state.shape == (2, 8)

    lstm.h = jnp.ones((2, 4))
    assert jnp.allclose(jnp.asarray(lstm.h), 1.0)
    assert jnp.allclose(jnp.asarray(lstm.c), 0.0)

    lstm.c = jnp.full((2, 4), 2.0)
    assert jnp.allclose(jnp.asarray(lstm.h), 1.0)
    assert jnp.allclose(jnp.asarray(lstm.c), 2.0)


def test_reservoir_bias_is_applied():
    """H-38: a nonzero bias shifts the reservoir output."""
    bm.random.seed(0)
    common = dict(in_connectivity=1.0, rec_connectivity=1.0,
                  activation_type='external', leaky_rate=1.0)
    r0 = bp.dyn.Reservoir(3, 5, b_initializer=init.ZeroInit(), **common)
    r1 = bp.dyn.Reservoir(3, 5, b_initializer=init.Constant(1.0), **common)
    # share the random weights so only the bias differs
    r1.Win.value = r0.Win.value
    r1.Wrec.value = r0.Wrec.value

    x = jnp.zeros((3,))
    out0 = jnp.asarray(r0.update(x))
    out1 = jnp.asarray(r1.update(x))
    # zero bias + zero input -> zero (external activation of 0 is tanh(0)=0)
    assert jnp.allclose(out0, 0.0)
    # nonzero bias shifts the output away from zero
    assert float(jnp.sum(jnp.abs(out1 - out0))) > 1e-3


def test_reservoir_recurrent_noise_is_symmetric():
    """H-37: recurrent noise is ``uniform(-1, 1)`` (zero-mean), not a -noise bias."""
    bm.random.seed(123)
    r = bp.dyn.Reservoir(2, 4000, noise_rec=1.0, in_connectivity=1.0,
                         rec_connectivity=1.0, activation_type='external',
                         leaky_rate=1.0)
    # zero out the weights so the state is driven purely by the recurrent noise
    r.Win.value = jnp.zeros_like(jnp.asarray(r.Win.value))
    r.Wrec.value = jnp.zeros_like(jnp.asarray(r.Wrec.value))
    out = jnp.asarray(r.update(jnp.zeros((2,))))
    # state = tanh(noise); symmetric noise -> mean near zero, both signs present
    assert abs(float(out.mean())) < 0.05
    assert float(out.min()) < -0.1
    assert float(out.max()) > 0.1


# ---------------------------------------------------------------------------
# Coverage tests: construct + step the assigned modules
# ---------------------------------------------------------------------------

def test_rate_populations_construct_and_step():
    """Cover FHN/FeedbackFHN/QIF/StuartLandau/WilsonCowan/ThresholdLinear."""
    bm.random.seed(0)
    _share()

    # FHN with OU noise enabled to exercise the noise branch; pass both inputs
    fhn = bp.dyn.FHN(3, x_ou_sigma=1.0, y_ou_sigma=1.0)
    fhn.reset_state()
    assert jnp.asarray(fhn.update(1.0, 0.5)).shape == (3,)
    fhn.clear_input()

    fbfhn = bp.dyn.FeedbackFHN(3, delay=2.0, x_ou_sigma=1.0, y_ou_sigma=1.0)
    fbfhn.reset_state()
    assert jnp.asarray(fbfhn.update(1.0, 0.5)).shape == (3,)
    fbfhn.clear_input()

    qif = bp.dyn.QIF(3)
    qif.reset_state()
    assert jnp.asarray(qif.update(1.0, 0.5)).shape == (3,)
    qif.clear_input()

    sl = bp.dyn.StuartLandauOscillator(3)
    sl.reset_state()
    assert jnp.asarray(sl.update(1.0, 0.5)).shape == (3,)
    sl.clear_input()

    wc = bp.dyn.WilsonCowanModel(3)
    wc.reset_state()
    assert jnp.asarray(wc.update(1.0, 0.5)).shape == (3,)
    wc.clear_input()

    tlm = bp.dyn.ThresholdLinearModel(3)
    tlm.reset_state()
    assert jnp.asarray(tlm.update(inp_e=1.0, inp_i=0.5)).shape == (3,)
    tlm.clear_input()


def test_rate_populations_no_input_var_branch():
    """Cover the ``input_var=False`` update branches with OU noise."""
    bm.random.seed(0)
    _share()
    for cls in (bp.dyn.FHN, bp.dyn.FeedbackFHN, bp.dyn.QIF,
                bp.dyn.StuartLandauOscillator, bp.dyn.WilsonCowanModel):
        m = cls(2, input_var=False, x_ou_sigma=1.0, y_ou_sigma=1.0)
        m.reset_state()
        out = m.update(1.0, 0.5)
        assert jnp.asarray(out).shape == (2,)


def test_rate_populations_run_via_dsrunner():
    """A short DSRunner run exercises update/clear_input across many steps."""
    bm.random.seed(0)
    fhn = bp.dyn.FHN(2)
    runner = bp.DSRunner(fhn, inputs=('input', 1.0), monitors=['x'],
                         progress_bar=False)
    runner.run(5.0)
    assert bool(jnp.isfinite(runner.mon['x']).all())


def test_reservoir_update_dense_and_sparse():
    """Cover Reservoir dense + sparse comp paths and feedforward noise."""
    bm.random.seed(0)
    # dense with feedforward noise + spectral radius scaling
    r = bp.dyn.Reservoir(4, 8, noise_in=0.1, spectral_radius=0.9)
    r.reset_state()
    out = r.update(jnp.ones((4,)))
    assert jnp.asarray(out).shape == (8,)

    # sparse computation path
    rs = bp.dyn.Reservoir(4, 8, comp_type='sparse', in_connectivity=0.5,
                          rec_connectivity=0.5)
    rs.reset_state()
    out_s = rs.update(jnp.ones((4,)))
    assert jnp.asarray(out_s).shape == (8,)


def test_rnn_cells_forward_unbatched():
    """Cover RNNCell/GRUCell/LSTMCell forward pass (unbatched)."""
    bm.random.seed(0)
    x = jnp.ones((3,))
    for cls in (bp.dyn.RNNCell, bp.dyn.GRUCell, bp.dyn.LSTMCell):
        c = cls(3, 4)
        out = c.update(x)
        assert jnp.asarray(out).shape == (4,)


def test_rnn_cells_forward_batched_and_train_state():
    """Cover batched forward + train_state initialization for the RNN cells."""
    bm.random.seed(0)
    x = jnp.ones((2, 3))
    for cls in (bp.dyn.RNNCell, bp.dyn.GRUCell, bp.dyn.LSTMCell):
        c = cls(3, 4, mode=bm.batching_mode)
        c.reset_state(2)
        out = c.update(x)
        assert jnp.asarray(out).shape == (2, 4)

        # train_state path
        ct = cls(3, 4, mode=bm.training_mode, train_state=True)
        ct.reset_state(2)
        out2 = ct.update(x)
        assert jnp.asarray(out2).shape == (2, 4)


def test_rnn_cells_no_bias_branch():
    """Cover the ``b_initializer=None`` (no bias) branches in the RNN cells."""
    bm.random.seed(0)
    x = jnp.ones((3,))
    for cls in (bp.dyn.RNNCell, bp.dyn.GRUCell, bp.dyn.LSTMCell):
        c = cls(3, 4, b_initializer=None)
        out = c.update(x)
        assert jnp.asarray(out).shape == (4,)


def test_dynold_compat_synapses_construct_and_step():
    """Cover dynold compat synapses: Exp/DualExp/Alpha (CUBA & COBA) + Delta."""
    import warnings
    bm.random.seed(0)
    bm.set_dt(0.1)

    from brainpy.synapses import (ExpCUBA, ExpCOBA, DualExpCUBA, DualExpCOBA,
                                  AlphaCUBA, AlphaCOBA, DeltaSynapse)

    factories = [
        lambda a, b: ExpCUBA(a, b, bp.connect.All2All(), tau=8.0),
        lambda a, b: ExpCOBA(a, b, bp.connect.All2All(), tau=8.0, E=0.0),
        lambda a, b: DualExpCUBA(a, b, bp.connect.All2All(), tau_decay=10.0, tau_rise=1.0),
        lambda a, b: DualExpCOBA(a, b, bp.connect.All2All(), tau_decay=10.0, tau_rise=1.0, E=0.0),
        lambda a, b: AlphaCUBA(a, b, bp.connect.All2All(), tau_decay=10.0),
        lambda a, b: AlphaCOBA(a, b, bp.connect.All2All(), tau_decay=10.0, E=0.0),
        lambda a, b: DeltaSynapse(a, b, bp.connect.All2All()),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for factory in factories:
            pre = bp.neurons.LIF(2)
            post = bp.neurons.LIF(2)
            syn = factory(pre, post)
            net = bp.Network(pre=pre, post=post, syn=syn)
            net.reset_state()
            _share()
            syn.update()  # must not raise


def test_dynold_stp_run_full_simulation():
    """Run the dynold STP learning rule via DSRunner to cover its update path."""
    bm.random.seed(0)
    bm.set_dt(0.1)
    pre = bp.neurons.LIF(1)
    post = bp.neurons.LIF(1)
    syn = bp.synapses.STP(pre, post, bp.connect.All2All(), U=0.2, tau_d=150.0, tau_f=2.0)
    net = bp.Network(pre=pre, syn=syn, post=post)
    runner = bp.DSRunner(net, inputs=[('pre.input', 28.0)],
                         monitors=['syn.I', 'syn.u', 'syn.x'], progress_bar=False)
    runner.run(30.0)
    assert bool(jnp.isfinite(runner.mon['syn.u']).all())
    assert bool(jnp.isfinite(runner.mon['syn.x']).all())


def test_reduced_models_construct_and_step():
    """Cover dynold reduced_models neurons: construct + a single update step."""
    from brainpy.dynold.neurons import reduced_models as rm

    bm.random.seed(0)
    bm.set_dt(0.1)
    _share()

    for cls in (rm.LeakyIntegrator, rm.LIF, rm.ExpIF, rm.AdExIF, rm.QuaIF,
                rm.AdQuaIF, rm.GIF, rm.Izhikevich, rm.HindmarshRose, rm.FHN,
                rm.ALIFBellec2020):
        m = cls(2)
        m.reset_state()
        out = m.update(1.0)
        assert jnp.asarray(out).shape == (2,)
        m.clear_input()


def test_reduced_models_no_input_var_branch():
    """Cover the ``input_var=False`` update branch of the reduced models."""
    from brainpy.dynold.neurons import reduced_models as rm

    bm.random.seed(0)
    bm.set_dt(0.1)
    _share()
    for cls in (rm.LIF, rm.ExpIF, rm.Izhikevich):
        m = cls(2, input_var=False)
        m.reset_state()
        out = m.update(1.0)
        assert jnp.asarray(out).shape == (2,)
        m.clear_input()


def test_reduced_models_tau_ref_and_noise_branches():
    """Cover the refractory + noise branches of the reduced models."""
    from brainpy.dynold.neurons import reduced_models as rm

    bm.random.seed(0)
    bm.set_dt(0.1)
    _share()

    # tau_ref + noise exercises the sdeint + refractory paths in *Ref models
    for cls in (rm.LIF, rm.ExpIF, rm.AdExIF, rm.QuaIF, rm.AdQuaIF,
                rm.Izhikevich, rm.GIF):
        m = cls(2, tau_ref=2.0, noise=0.5)
        m.reset_state()
        out = m.update(5.0)
        assert jnp.asarray(out).shape == (2,)
        m.clear_input()


def test_bellec_sfa_models_construct_and_step():
    """Cover ALIFBellec2020 / LIF_SFA_Bellec2020 with and without refractoriness."""
    from brainpy.dynold.neurons import reduced_models as rm

    bm.random.seed(0)
    bm.set_dt(0.1)
    _share()

    for cls in (rm.ALIFBellec2020, rm.LIF_SFA_Bellec2020):
        m = cls(2)
        m.reset_state()
        assert jnp.asarray(m.update(5.0)).shape == (2,)

        m_ref = cls(2, tau_ref=2.0)
        m_ref.reset_state()
        assert jnp.asarray(m_ref.update(5.0)).shape == (2,)


def test_conv_lstm_cells_forward():
    """Cover the convolutional LSTM cells (1d/2d/3d) forward pass."""
    bm.random.seed(0)

    c1 = bp.dyn.Conv1dLSTMCell(input_shape=(5,), in_channels=2, out_channels=3,
                               kernel_size=3, mode=bm.batching_mode)
    c1.reset_state(1)
    assert jnp.asarray(c1.update(jnp.ones((1, 5, 2)))).shape == (1, 5, 3)

    c2 = bp.dyn.Conv2dLSTMCell(input_shape=(4, 4), in_channels=2, out_channels=3,
                               kernel_size=3, mode=bm.batching_mode)
    c2.reset_state(1)
    assert jnp.asarray(c2.update(jnp.ones((1, 4, 4, 2)))).shape == (1, 4, 4, 3)


def test_poisson_input_binomial_branch_and_helpers():
    """Cover the small-N binomial branch, ``__repr__`` and ``reset`` of PoissonInput."""
    from brainpy.dynold.experimental.others import PoissonInput

    bm.random.seed(0)
    bm.set_dt(0.1)
    _share()

    # low freq / small num_input -> a<=5 or b<=5 -> binomial branch
    pi = PoissonInput(target_shape=(5,), num_input=10, freq=10.0, weight=1.0)
    out = pi.update()
    assert jnp.asarray(out).shape == (5,)
    assert 'PoissonInput' in repr(pi)
    pi.reset()
    pi.reset_state()
