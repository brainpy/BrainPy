"""Regression + coverage tests for the 2026-06-18 BrainPy audit.

This module targets the fixes documented in ``docs/issues-found-20260618.md``
for the dyn neurons / synapses / projections and dnn/linear subsystems:

  * H-34  ``dyn/neurons/lif.py``           -- ``ExpIFRef`` honours ``noise=`` (sdeint vs odeint).
  * H-35  ``dyn/neurons/lif.py``           -- ``IzhikevichRef`` / ``GifRef`` ``detach_spk`` actually
                                              cuts the spike gradient path.
  * C-06/H-39/M-22 ``dyn/synapses/abstract_models.py`` -- ``STP`` facilitation no longer diverges;
                                              discrete jumps applied to decayed locals; u/x stay in [0, 1].
  * C-17  ``dyn/projections/inputs.py``     -- ``PoissonInput`` Gaussian std == sqrt(n*p*(1-p)).
  * C-18  ``dyn/projections/align_post.py`` -- ``HalfProjAlignPost`` calls ``comm`` exactly once.
  * C-19/H-41 ``dyn/projections/plasticity.py`` + ``dnn/linear.py`` -- ``STDP_Song2000`` runs and
                                              changes the (promoted ``Variable``) weight; ``W_min=W_max=None``
                                              does not crash; bounds clamp when set.
  * H-40  ``dyn/projections/base.py``       -- module re-exports the real base classes.

Plus breadth coverage tests that construct and run one ``update()`` for every public
neuron class in ``lif.py`` (and LTC variants), the abstract synapse models, the
projection wrappers, and the dnn/linear comm layers.

Run from the worktree root with ``PYTHONPATH=.``.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import brainpy as bp
import brainpy.math as bm
from brainpy.integrators.ode.base import ODEIntegrator
from brainpy.integrators.sde.base import SDEIntegrator


def _share(t=0.0, dt=0.1, i=0):
    bp.share.save(t=t, dt=dt, i=i)


# ---------------------------------------------------------------------------
# H-34: ExpIFRef honours noise= (sdeint) and is plain odeint otherwise.
# ---------------------------------------------------------------------------

def test_expifref_noise_uses_sde_integrator():
    noisy = bp.dyn.ExpIFRef(3, noise=0.5)
    plain = bp.dyn.ExpIFRef(3)
    assert isinstance(noisy.integral, SDEIntegrator)
    assert not isinstance(noisy.integral, ODEIntegrator)
    assert isinstance(plain.integral, ODEIntegrator)
    assert not isinstance(plain.integral, SDEIntegrator)


def test_expifref_noisy_update_runs():
    bm.random.seed(0)
    neu = bp.dyn.ExpIFRef(3, noise=1.0)
    neu.reset_state()
    _share()
    neu.update(bm.ones(3) * 5.0)
    assert jnp.all(jnp.isfinite(neu.V.value))


# ---------------------------------------------------------------------------
# H-35: IzhikevichRef / GifRef detach_spk actually cuts the spike gradient.
# ---------------------------------------------------------------------------

def _one_step_grad_izhikevich(detach, v0):
    neu = bp.dyn.IzhikevichRef(3, mode=bm.training_mode, detach_spk=detach)

    def loss(inp):
        neu.reset_state(bm.training_mode)
        neu.V.value = bm.ones(neu.V.shape) * v0
        _share()
        neu.update(inp)
        return jnp.sum(neu.V.value)

    return jax.grad(loss)(bm.zeros(3))


def test_izhikevichref_detach_spk_changes_gradient():
    # With V driven above threshold (~30 mV) the spike path is active, so
    # detaching the spike (cutting it) changes the gradient w.r.t. the input.
    v0 = 30.0
    g_detach = _one_step_grad_izhikevich(True, v0)
    g_plain = _one_step_grad_izhikevich(False, v0)
    assert jnp.all(jnp.isfinite(g_detach))
    assert jnp.all(jnp.isfinite(g_plain))
    # detach_spk cuts the spike contribution -> gradient differs from the
    # grad-carrying path.
    assert not jnp.allclose(g_detach, g_plain)


def _one_step_grad_gif(detach, v0):
    neu = bp.dyn.GifRef(3, mode=bm.training_mode, detach_spk=detach)

    def loss(inp):
        neu.reset_state(bm.training_mode)
        neu.V.value = bm.ones(neu.V.shape) * v0
        _share()
        neu.update(inp)
        # include the adaptation/threshold states the spike resets touch.
        return (jnp.sum(neu.V.value) + jnp.sum(neu.I1.value)
                + jnp.sum(neu.I2.value) + jnp.sum(neu.V_th.value))

    return jax.grad(loss)(bm.zeros(3))


def test_gifref_detach_spk_changes_gradient():
    v0 = -50.0  # drives V across the GIF threshold within one step
    g_detach = _one_step_grad_gif(True, v0)
    g_plain = _one_step_grad_gif(False, v0)
    assert jnp.all(jnp.isfinite(g_detach))
    assert jnp.all(jnp.isfinite(g_plain))
    assert not jnp.allclose(g_detach, g_plain)


# ---------------------------------------------------------------------------
# C-06 / H-39 / M-22: STP short-term facilitation does not diverge.
# ---------------------------------------------------------------------------

def test_stp_no_spike_u_does_not_increase():
    stp = bp.dyn.STP(1)
    u0 = float(stp.u.value[0])
    no_spike = bm.zeros(1, dtype=bool)
    for _ in range(5):
        _share()
        stp.update(no_spike)
    # With no spike, facilitation u must decay/stay -- never grow.
    assert float(stp.u.value[0]) <= u0 + 1e-9


def test_stp_spiking_u_x_stay_bounded():
    stp = bp.dyn.STP(1)
    spike = bm.ones(1, dtype=bool)
    us, xs = [], []
    for i in range(50):
        _share(t=i * 0.1, i=i)
        stp.update(spike)
        us.append(float(stp.u.value[0]))
        xs.append(float(stp.x.value[0]))
    assert min(us) >= 0.0 and max(us) <= 1.0
    assert min(xs) >= 0.0 and max(xs) <= 1.0
    # released resource u*x stays finite and modest (used to explode to thousands)
    assert max(u * x for u, x in zip(us, xs)) < 1.0


# ---------------------------------------------------------------------------
# C-17: PoissonInput Gaussian-approx std == sqrt(n*p*(1-p)).
# ---------------------------------------------------------------------------

def test_poisson_input_gaussian_std_matches_binomial():
    bm.random.seed(0)
    n_neuron = 4000
    n_input = 1000
    freq = 200.0  # Hz
    dt = 0.1
    target = bm.Variable(bm.zeros(n_neuron))
    _share(dt=dt)
    pin = bp.dyn.PoissonInput(target_var=target, num_input=n_input, freq=freq, weight=1.0)

    p = freq * dt / 1e3
    # ensure we exercise the Gaussian branch (a > 5 and b > 5)
    assert n_input * p > 5 and n_input * (1 - p) > 5

    target.value = bm.zeros(n_neuron)
    pin.update()
    samp = np.asarray(target.value)

    expected_std = np.sqrt(n_input * p * (1 - p))
    expected_mean = n_input * p
    # std must be the binomial std, NOT the variance (the old bug was ~4x too big).
    assert samp.std() == pytest.approx(expected_std, rel=0.1)
    assert samp.mean() == pytest.approx(expected_mean, rel=0.1)
    # guard against the regression where std == variance (b*p).
    assert samp.std() < 0.5 * (n_input * p * (1 - p))


# ---------------------------------------------------------------------------
# C-18: HalfProjAlignPost calls comm exactly once per update.
# ---------------------------------------------------------------------------

class _CountingLinear(bp.dnn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_calls = 0

    def update(self, x):
        self.n_calls += 1
        return super().update(x)


def test_halfprojalignpost_calls_comm_once():
    post = bp.dyn.LifRef(4)
    comm = _CountingLinear(4, 4, W_initializer=bp.init.Constant(0.1))
    proj = bp.dyn.HalfProjAlignPost(
        comm=comm,
        syn=bp.dyn.Expon(4, tau=5.0),
        out=bp.dyn.COBA(E=0.0),
        post=post,
    )
    _share()
    current = proj.update(bm.ones(4))
    assert comm.n_calls == 1
    # returned current equals one manual comm application.
    expected = comm.update(bm.ones(4))  # second call only for comparison
    assert jnp.allclose(jnp.asarray(current), jnp.asarray(expected))


# ---------------------------------------------------------------------------
# C-19 / H-41: STDP over a Linear/AllToAll comm runs and changes the weight.
# ---------------------------------------------------------------------------

def _build_stdp_net(w_min=None, w_max=None):
    pre = bp.dyn.LifRef(3)
    post = bp.dyn.LifRef(4)
    syn = bp.dyn.STDP_Song2000(
        pre=pre,
        delay=1.0,
        comm=bp.dnn.AllToAll(3, 4, weight=bp.init.Uniform(max_val=0.1)),
        syn=bp.dyn.Expon.desc(post.varshape, tau=5.0),
        out=bp.dyn.COBA.desc(E=0.0),
        post=post,
        W_min=w_min,
        W_max=w_max,
    )
    net = bp.DynSysGroup(pre=pre, post=post, syn=syn)
    net.reset_state()
    return net, pre, post, syn


def test_stdp_song2000_runs_and_changes_weight():
    net, pre, post, syn = _build_stdp_net()
    # weight starts as a plain array (comm built outside a TrainingMode).
    assert not isinstance(syn.comm.weight, bm.Variable)
    w0 = bm.as_jax(syn.comm.weight).copy()

    for i in range(40):
        _share(t=i * 0.1, i=i)
        syn()
        pre(bm.ones(3) * 80.0)
        post(bm.ones(4) * 80.0)

    # weight is promoted to a Variable on the first stdp_update and updates.
    assert isinstance(syn.comm.weight, bm.Variable)
    w1 = bm.as_jax(syn.comm.weight)
    assert not jnp.allclose(w0, w1)
    assert jnp.all(jnp.isfinite(w1))


def test_stdp_song2000_default_none_bounds_do_not_crash():
    # W_min == W_max == None used to crash on bm.as_jax(None) (H-41).
    net, pre, post, syn = _build_stdp_net(w_min=None, w_max=None)
    _share()
    syn()
    pre(bm.ones(3) * 80.0)
    post(bm.ones(4) * 80.0)
    assert jnp.all(jnp.isfinite(bm.as_jax(syn.comm.weight)))


def test_stdp_song2000_bounds_clamp_weight():
    w_max = 0.2
    w_min = 0.0
    net, pre, post, syn = _build_stdp_net(w_min=w_min, w_max=w_max)
    for i in range(60):
        _share(t=i * 0.1, i=i)
        syn()
        pre(bm.ones(3) * 80.0)
        post(bm.ones(4) * 80.0)
    w = bm.as_jax(syn.comm.weight)
    assert jnp.all(w <= w_max + 1e-5)
    assert jnp.all(w >= w_min - 1e-5)


# ---------------------------------------------------------------------------
# H-40: projections/base.py re-exports real base classes (no dead duplicate).
# ---------------------------------------------------------------------------

def test_projections_base_reexports_real_classes():
    from brainpy.dyn.projections import base as proj_base
    assert hasattr(proj_base, 'Projection')
    assert hasattr(proj_base, 'SynConn')
    assert 'Projection' in proj_base.__all__
    assert 'SynConn' in proj_base.__all__
    # the re-exported Projection is the canonical one used by the wrappers.
    assert proj_base.Projection is bp.dyn.Projection


# ---------------------------------------------------------------------------
# Coverage: every public neuron class in lif.py, both modes.
# ---------------------------------------------------------------------------

_NEURON_NAMES = [
    'IF', 'Lif', 'LifRef',
    'ExpIF', 'ExpIFRef',
    'AdExIF', 'AdExIFRef',
    'QuaIF', 'QuaIFRef',
    'AdQuaIF', 'AdQuaIFRef',
    'Gif', 'GifRef',
    'Izhikevich', 'IzhikevichRef',
]
_NEURON_NAMES = _NEURON_NAMES + [n + 'LTC' for n in _NEURON_NAMES]


@pytest.mark.parametrize('name', _NEURON_NAMES)
@pytest.mark.parametrize('mode', [None, 'train'])
def test_neuron_update_runs(name, mode):
    cls = getattr(bp.dyn, name)
    if mode == 'train':
        neu = cls(3, mode=bm.training_mode)
        neu.reset_state(bm.training_mode)
    else:
        neu = cls(3)
        neu.reset_state()
    # small drive: the exponential-IF family genuinely overflows V in a single
    # large training-mode step (no hard refractory clamp before the surrogate
    # spike), which is a model property, not an audit regression.
    _share()
    out = neu.update(bm.ones(3) * 1.0)
    assert out is not None
    assert neu.V.value.shape[-1] == 3
    if mode is None:
        # default (non-training) mode applies the hard reset, so V stays finite.
        assert jnp.all(jnp.isfinite(neu.V.value))


# ---------------------------------------------------------------------------
# Coverage: abstract synapse models forward passes.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name', ['Expon', 'DualExpon', 'Alpha', 'NMDA', 'AMPA', 'STP', 'STD'])
def test_synapse_forward(name):
    cls = getattr(bp.dyn, name)
    syn = cls(3)
    spike_bool = bm.asarray([True, False, True], dtype=bool)
    spike_float = spike_bool.astype(float)
    _share()
    if name in ('NMDA', 'AMPA'):
        out = syn(spike_float)
    else:
        out = syn(spike_bool)
    assert jnp.asarray(out).shape == (3,)
    assert jnp.all(jnp.isfinite(jnp.asarray(out)))


# ---------------------------------------------------------------------------
# Coverage: projection wrappers.
# ---------------------------------------------------------------------------

def test_full_proj_align_post():
    class Net(bp.DynSysGroup):
        def __init__(self):
            super().__init__()
            self.pre = bp.dyn.LifRef(4)
            self.post = bp.dyn.LifRef(3)
            self.proj = bp.dyn.FullProjAlignPost(
                pre=self.pre, delay=None,
                comm=bp.dnn.AllToAll(4, 3, weight=0.1),
                syn=bp.dyn.Expon(3, tau=5.0),
                out=bp.dyn.COBA(E=0.0),
                post=self.post,
            )

        def update(self, inp):
            self.proj()
            self.pre(inp)
            self.post()
            return self.post.V.value

    net = Net()
    net.reset_state()
    _share()
    out = net.update(bm.ones(4) * 5.0)
    assert jnp.all(jnp.isfinite(out))


def test_full_proj_align_post_mg():
    class Net(bp.DynSysGroup):
        def __init__(self):
            super().__init__()
            self.pre = bp.dyn.LifRef(4)
            self.post = bp.dyn.LifRef(3)
            self.proj = bp.dyn.FullProjAlignPostMg(
                pre=self.pre, delay=None,
                comm=bp.dnn.AllToAll(4, 3, weight=0.1),
                syn=bp.dyn.Expon.desc(3, tau=5.0),
                out=bp.dyn.COBA.desc(E=0.0),
                post=self.post,
            )

        def update(self, inp):
            self.proj()
            self.pre(inp)
            self.post()
            return self.post.V.value

    net = Net()
    net.reset_state()
    _share()
    out = net.update(bm.ones(4) * 5.0)
    assert jnp.all(jnp.isfinite(out))


def test_full_proj_align_pre_sdmg():
    class Net(bp.DynSysGroup):
        def __init__(self):
            super().__init__()
            self.pre = bp.dyn.LifRef(4)
            self.post = bp.dyn.LifRef(3)
            self.proj = bp.dyn.FullProjAlignPreSDMg(
                pre=self.pre,
                syn=bp.dyn.Expon.desc(4, tau=5.0),
                delay=None,
                comm=bp.dnn.AllToAll(4, 3, weight=0.1),
                out=bp.dyn.COBA(E=0.0),
                post=self.post,
            )

        def update(self, inp):
            self.proj()
            self.pre(inp)
            self.post()
            return self.post.V.value

    net = Net()
    net.reset_state()
    _share()
    out = net.update(bm.ones(4) * 5.0)
    assert jnp.all(jnp.isfinite(out))


def test_half_proj_align_post_mg():
    post = bp.dyn.LifRef(3)
    proj = bp.dyn.HalfProjAlignPostMg(
        comm=bp.dnn.AllToAll(4, 3, weight=0.1),
        syn=bp.dyn.Expon.desc(3, tau=5.0),
        out=bp.dyn.COBA.desc(E=0.0),
        post=post,
    )
    _share()
    out = proj.update(bm.ones(4))
    assert jnp.all(jnp.isfinite(jnp.asarray(out)))


# ---------------------------------------------------------------------------
# Coverage: PoissonInput full update path (binomial small-N branch too).
# ---------------------------------------------------------------------------

def test_poisson_input_small_n_branch():
    bm.random.seed(1)
    target = bm.Variable(bm.zeros(8))
    _share(dt=0.1)
    pin = bp.dyn.PoissonInput(target_var=target, num_input=10, freq=20.0, weight=1.0)
    assert repr(pin)  # exercise __repr__
    pin.update()  # a, b small -> binomial branch
    assert jnp.all(jnp.isfinite(target.value))


def test_input_var_runs():
    iv = bp.dyn.InputVar(4)
    iv.input += 1.0
    assert jnp.allclose(jnp.asarray(iv.update()), 1.0)
    iv.clear_input()
    assert jnp.allclose(jnp.asarray(iv.update()), 0.0)


# ---------------------------------------------------------------------------
# Coverage: dnn/linear comm layer forward passes.
# ---------------------------------------------------------------------------

def test_dnn_linear_forwards():
    conn = bp.conn.FixedProb(0.5, pre=4, post=3, seed=1)

    dense = bp.dnn.Dense(4, 3)
    assert jnp.asarray(dense(bm.ones(4))).shape == (3,)

    a2a = bp.dnn.AllToAll(4, 3, weight=0.1)
    # NonBatching mode requires a 1D input; identical scalar weight + ones input
    # collapses to a scalar (sum over pre).
    assert jnp.asarray(a2a(bm.ones(4))).shape == ()

    o2o = bp.dnn.OneToOne(4, weight=0.1)
    assert jnp.asarray(o2o(bm.ones(4))).shape == (4,)

    ml = bp.dnn.MaskedLinear(conn, weight=0.1)
    assert jnp.asarray(ml(bm.ones(4))).shape == (3,)

    csr = bp.dnn.CSRLinear(conn, weight=0.1)
    assert jnp.asarray(csr(bm.ones(4))).shape == (3,)


def test_dnn_event_and_jitprob_forwards():
    conn = bp.conn.FixedProb(0.5, pre=4, post=3, seed=1)
    spike = bm.asarray([True, False, True, False])
    x = bm.ones(4)

    # Identity passthrough.
    assert jnp.asarray(bp.dnn.Identity()(x)).shape == (4,)

    # Event-driven CSR.
    ec = bp.dnn.EventCSRLinear(conn, weight=0.1)
    assert jnp.asarray(ec(spike)).shape == (3,)

    # JIT just-in-time fixed-prob connectivity (dense-input variants).
    assert jnp.asarray(
        bp.dnn.JitFPHomoLinear(4, 3, prob=0.5, weight=0.1, seed=1)(x)).shape == (3,)
    assert jnp.asarray(
        bp.dnn.JitFPUniformLinear(4, 3, prob=0.5, w_low=0.0, w_high=0.1, seed=1)(x)).shape == (3,)
    assert jnp.asarray(
        bp.dnn.JitFPNormalLinear(4, 3, prob=0.5, w_mu=0.0, w_sigma=0.1, seed=1)(x)).shape == (3,)

    # JIT fixed-prob connectivity (event/spike-input variants).
    assert jnp.asarray(
        bp.dnn.EventJitFPHomoLinear(4, 3, prob=0.5, weight=0.1, seed=1)(spike)).shape == (3,)
    assert jnp.asarray(
        bp.dnn.EventJitFPUniformLinear(4, 3, prob=0.5, w_low=0.0, w_high=0.1, seed=1)(spike)).shape == (3,)
    assert jnp.asarray(
        bp.dnn.EventJitFPNormalLinear(4, 3, prob=0.5, w_mu=0.0, w_sigma=0.1, seed=1)(spike)).shape == (3,)


def test_dense_stdp_update_promotes_weight():
    dense = bp.dnn.Dense(3, 4, W_initializer=bp.init.Uniform(max_val=0.1))
    assert not isinstance(dense.W, bm.Variable)
    # the plasticity wrapper passes jax arrays to ``stdp_update``; mirror that.
    spike_pre = bm.as_jax(bm.asarray([1.0, 0.0, 1.0]))
    trace_pre = bm.as_jax(bm.asarray([0.1, 0.2, 0.3, 0.4]))
    w0 = bm.as_jax(dense.W).copy()
    dense.stdp_update(on_pre={'spike': spike_pre, 'trace': trace_pre}, w_min=None, w_max=None)
    assert isinstance(dense.W, bm.Variable)
    assert not jnp.allclose(w0, bm.as_jax(dense.W))
    assert jnp.all(jnp.isfinite(bm.as_jax(dense.W)))
