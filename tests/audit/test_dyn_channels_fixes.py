"""Regression + coverage tests for the BrainPy v2.7.8 audit (2026-06-18).

This module pins the channel/ion fixes documented in ``docs/issues-found-20260618.md``:

* **C-14** -- Standalone HH/Markov channel gating produced NaN at the voltage
  singularities (``channels/sodium.py``, ``potassium.py``, ``calcium.py`` and the
  ``*_compatible`` legacy duplicates). The rate functions of the form
  ``k * temp / (1 - exp(-temp / d))`` are ``0/0`` exactly at the removable
  singularity.  After the fix they are rewritten with a branch-safe ``exprel``
  helper so that the value *and* its gradient are finite there.
  Audit repro: ``IK_HH1952v2(1).f_p_alpha([-55.0])`` returns ~0.1, not nan.
* **M-17** -- ``PotassiumFixed`` default ``E`` was ``-950`` mV (typo); fixed to
  ``-95`` mV.
* **H-33** -- ``dyn/ions/base.py`` registered every channel under the literal name
  ``"k"`` (``self.add_elem(k=v)``), so channels overwrote each other.  The fix
  (``self.add_elem(**{k: v})``) registers each channel under its real name.

The remaining tests build every public channel class against a Hodgkin-Huxley
style host and exercise ``reset_state`` / ``update`` for coverage.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainpy as bp
import brainpy.math as bm


# Voltage sweep that intentionally includes the removable-singularity voltages.
_V_SWEEP = jnp.asarray(np.linspace(-120.0, 60.0, 91), dtype=bm.float_)


def _assert_finite_with_grad(make_channel, fname, singular_v):
    """A rate function and its gradient must be finite across a sweep incl. the singular V."""
    chan = make_channel()
    f = getattr(chan, fname)

    # Value finite across the whole sweep (which contains the singular voltage).
    vals = np.asarray(f(_V_SWEEP))
    assert np.all(np.isfinite(vals)), f"{fname}: non-finite value over sweep: {vals}"

    # Value finite exactly at the singular voltage.
    sval = np.asarray(f(jnp.asarray([singular_v], dtype=bm.float_)))
    assert np.all(np.isfinite(sval)), f"{fname}: non-finite value at singular V={singular_v}: {sval}"

    # Gradient finite at the singular voltage (the part bm.where clamping cannot fix).
    def scalar(v):
        return getattr(make_channel(), fname)(jnp.asarray([v], dtype=bm.float_))[0]

    g = float(jax.grad(scalar)(float(singular_v)))
    assert np.isfinite(g), f"{fname}: non-finite gradient at singular V={singular_v}: {g}"


# ---------------------------------------------------------------------------
# C-14 regression: finite value + finite gradient at the singular voltages.
# Each tuple is (class, rate-function name, singular voltage).
# The singular voltage is where the argument of the exprel helper is zero.
# ---------------------------------------------------------------------------

# Legacy/compatible exported classes (master_type = HHTypedNeuron).
_LEGACY_SINGULAR_CASES = [
    # IKDR_Ba2002 V_sh=-50: alpha singular at V - V_sh - 15 = 0 -> V = -35
    ("IKDR_Ba2002", "f_p_alpha", -35.0),
    # IK_TM1991 V_sh=-60: alpha singular at 15 - V + V_sh = 0 -> V = -45
    ("IK_TM1991", "f_p_alpha", -45.0),
    # IK_HH1952 V_sh=-45: alpha singular at V - V_sh + 10 = 0 -> V = -55
    ("IK_HH1952", "f_p_alpha", -55.0),
    # INa_Ba2002 V_sh=-50: p_alpha V-V_sh-13=0 -> -37 ; p_beta V-V_sh-40=0 -> -10
    ("INa_Ba2002", "f_p_alpha", -37.0),
    ("INa_Ba2002", "f_p_beta", -10.0),
    # INa_TM1991 V_sh=-63: p_alpha 13-V+V_sh=0 -> -50 ; p_beta V-V_sh-40=0 -> -23
    ("INa_TM1991", "f_p_alpha", -50.0),
    ("INa_TM1991", "f_p_beta", -23.0),
    # INa_HH1952 V_sh=-45: p_alpha V-V_sh-5=0 -> -40
    ("INa_HH1952", "f_p_alpha", -40.0),
]

# v2 classes (master_type = an Ion subtype).
_V2_SINGULAR_CASES = [
    ("IKDR_Ba2002v2", "f_p_alpha", -35.0),
    ("IK_TM1991v2", "f_p_alpha", -45.0),
    ("IK_HH1952v2", "f_p_alpha", -55.0),
    ("INa_Ba2002v2", "f_p_alpha", -37.0),
    ("INa_Ba2002v2", "f_p_beta", -10.0),
    ("INa_TM1991v2", "f_p_alpha", -50.0),
    ("INa_TM1991v2", "f_p_beta", -23.0),
    ("INa_HH1952v2", "f_p_alpha", -40.0),
    # ICaHT_Re1993 (markov v2, calcium) V_sh=0: p_alpha -27-V+V_sh=0 -> V=-27
    ("ICaHT_Re1993", "f_p_alpha", -27.0),
]


@pytest.mark.parametrize("clsname,fname,singular_v", _LEGACY_SINGULAR_CASES)
def test_c14_legacy_rate_finite_and_grad(clsname, fname, singular_v):
    cls = getattr(bp.dyn, clsname)
    _assert_finite_with_grad(lambda: cls(1), fname, singular_v)


@pytest.mark.parametrize("clsname,fname,singular_v", _V2_SINGULAR_CASES)
def test_c14_v2_rate_finite_and_grad(clsname, fname, singular_v):
    cls = getattr(bp.dyn, clsname)
    _assert_finite_with_grad(lambda: cls(1), fname, singular_v)


def test_c14_ik_hh1952v2_audit_repro():
    """Exact audit repro: was [nan], must now be ~0.1."""
    val = np.asarray(bp.dyn.IK_HH1952v2(1).f_p_alpha(jnp.asarray([-55.0], dtype=bm.float_)))
    assert np.all(np.isfinite(val))
    assert np.allclose(val, 0.1, atol=1e-4)


def test_c14_all_defined_rate_functions_finite_over_sweep():
    """Sweep every rate function each markov/ss channel defines; none may produce NaN."""
    classes = [
        bp.dyn.IKDR_Ba2002, bp.dyn.IK_TM1991, bp.dyn.IK_HH1952,
        bp.dyn.INa_Ba2002, bp.dyn.INa_TM1991, bp.dyn.INa_HH1952,
        bp.dyn.IKDR_Ba2002v2, bp.dyn.IK_TM1991v2, bp.dyn.IK_HH1952v2,
        bp.dyn.INa_Ba2002v2, bp.dyn.INa_TM1991v2, bp.dyn.INa_HH1952v2,
        bp.dyn.ICaHT_Re1993,
    ]
    for cls in classes:
        chan = cls(1)
        for fname in ("f_p_alpha", "f_p_beta", "f_q_alpha", "f_q_beta"):
            f = getattr(chan, fname, None)
            if f is None:
                continue
            try:
                vals = np.asarray(f(_V_SWEEP))
            except NotImplementedError:
                continue
            assert np.all(np.isfinite(vals)), f"{cls.__name__}.{fname} produced NaN/inf"


# ---------------------------------------------------------------------------
# M-17 regression: PotassiumFixed default reversal potential.
# ---------------------------------------------------------------------------

def test_m17_potassium_fixed_default_E():
    ion = bp.dyn.PotassiumFixed(1)
    assert np.allclose(np.asarray(ion.E), -95.0), f"expected -95 mV, got {ion.E}"
    # And not the historical buggy value.
    assert not np.allclose(np.asarray(ion.E), -950.0)


# ---------------------------------------------------------------------------
# H-33 regression: MixIons / Ion.add_elem must register channels under their
# real names instead of collapsing them all under the literal key "k".
# ---------------------------------------------------------------------------

def test_h33_ion_add_elem_keeps_distinct_names():
    na = bp.dyn.SodiumFixed(2, E=50.)
    ina_hh = bp.dyn.INa_HH1952v2(2)
    ina_ba = bp.dyn.INa_Ba2002v2(2)
    na.add_elem(ina_hh=ina_hh, ina_ba=ina_ba)

    # Both channels stored under their real names, none lost to a literal "k".
    assert set(na.children.keys()) == {"ina_hh", "ina_ba"}
    assert "k" not in na.children
    assert na.children["ina_hh"] is ina_hh
    assert na.children["ina_ba"] is ina_ba
    assert ina_hh.name != ina_ba.name


def test_h33_mixions_registers_distinct_multiion_channels():
    """Build a MixIons over a Calcium+Potassium JointType channel (IAHP_De1994v2).

    Two distinct channels must coexist under their real names; before the fix
    the second would overwrite the first under the key "k".
    """
    bm.set_dt(0.1)

    class Net(bp.dyn.CondNeuGroupLTC):
        def __init__(self, size):
            super().__init__(size)
            self.Ca = bp.dyn.CalciumDetailed(size)
            self.K = bp.dyn.PotassiumFixed(size, E=-95.)
            self.KCa = bp.dyn.MixIons(self.Ca, self.K)
            self.KCa.add_elem(iahp1=bp.dyn.IAHP_De1994v2(size),
                              iahp2=bp.dyn.IAHP_De1994v2(size))

    net = Net(2)
    net.reset_state()
    assert set(net.KCa.children.keys()) == {"iahp1", "iahp2"}
    assert "k" not in net.KCa.children

    bp.share.save(t=0., dt=0.1, i=0)
    net.update(bm.ones(2) * -50.)
    cur = np.asarray(net.KCa.current(net.V.value))
    assert np.all(np.isfinite(cur))


def test_h33_mix_ions_helper_and_ion_current_paths():
    bm.set_dt(0.1)
    na = bp.dyn.SodiumFixed(2, E=50.)
    na.add_elem(ina=bp.dyn.INa_HH1952v2(2))
    k = bp.dyn.PotassiumFixed(2, E=-95.)
    k.add_elem(ik=bp.dyn.IK_HH1952v2(2))

    mix = bp.dyn.mix_ions(na, k)
    assert isinstance(mix, bp.dyn.MixIons)

    V = bm.ones(2) * -65.
    na.reset_state(V)
    k.reset_state(V)
    # Exercise Ion.update / Ion.current / pack_info on ions/base.py + ions/potassium.py.
    bp.share.save(t=0., dt=0.1, i=0)
    na.update(V)
    k.update(V)
    assert np.all(np.isfinite(np.asarray(k.current(V))))
    assert set(k.pack_info().keys()) == {"C", "E"}


# ---------------------------------------------------------------------------
# Coverage: construct & exercise every public channel against an HH-style host.
# ---------------------------------------------------------------------------

def _run_cond_neu_group(net, n_steps=2):
    net.reset_state()
    bp.share.save(t=0., dt=bm.get_dt(), i=0)
    for i in range(n_steps):
        bp.share.save(t=i * bm.get_dt(), i=i)
        net.update(bm.ones(net.num) * 1.0)


def test_coverage_compatible_potassium_and_sodium_channels():
    """Legacy/compatible channels are hosted directly by a CondNeuGroup (HH neuron)."""
    bm.set_dt(0.1)

    class Net(bp.dyn.CondNeuGroup):
        def __init__(self, size):
            super().__init__(size)
            # sodium_compatible.py
            self.INa_Ba = bp.dyn.INa_Ba2002(size)
            self.INa_TM = bp.dyn.INa_TM1991(size)
            self.INa_HH = bp.dyn.INa_HH1952(size)
            # potassium_compatible.py
            self.IKDR = bp.dyn.IKDR_Ba2002(size)
            self.IK_TM = bp.dyn.IK_TM1991(size)
            self.IK_HH = bp.dyn.IK_HH1952(size)
            self.IKA1 = bp.dyn.IKA1_HM1992(size)
            self.IKA2 = bp.dyn.IKA2_HM1992(size)
            self.IKK2A = bp.dyn.IKK2A_HM1992(size)
            self.IKK2B = bp.dyn.IKK2B_HM1992(size)
            self.IKNI = bp.dyn.IKNI_Ya1989(size)
            self.IKL = bp.dyn.IKL(size)

    _run_cond_neu_group(Net(2))


def test_coverage_v2_sodium_and_potassium_channels():
    """v2 channels (potassium.py / sodium.py) are hosted by Sodium/Potassium ions."""
    bm.set_dt(0.1)

    class Net(bp.dyn.CondNeuGroupLTC):
        def __init__(self, size):
            super().__init__(size)
            self.Na = bp.dyn.SodiumFixed(size, E=50.)
            self.Na.add_elem(
                ina_ba=bp.dyn.INa_Ba2002v2(size),
                ina_tm=bp.dyn.INa_TM1991v2(size),
                ina_hh=bp.dyn.INa_HH1952v2(size),
            )
            self.K = bp.dyn.PotassiumFixed(size, E=-95.)
            self.K.add_elem(
                ikdr=bp.dyn.IKDR_Ba2002v2(size),
                ik_tm=bp.dyn.IK_TM1991v2(size),
                ik_hh=bp.dyn.IK_HH1952v2(size),
                ika1=bp.dyn.IKA1_HM1992v2(size),
                ika2=bp.dyn.IKA2_HM1992v2(size),
                ikk2a=bp.dyn.IKK2A_HM1992v2(size),
                ikk2b=bp.dyn.IKK2B_HM1992v2(size),
                ikni=bp.dyn.IKNI_Ya1989v2(size),
                ik_leak=bp.dyn.IK_Leak(size),
            )

    _run_cond_neu_group(Net(2))


def test_coverage_potassium_module_legacy_classes():
    """potassium.py also ships HHTypedNeuron-hosted legacy duplicates that are
    shadowed at the ``bp.dyn`` level by potassium_compatible.py.  Import them
    directly from the module so the in-file C-14 fixes are exercised too.
    """
    import brainpy.dyn.channels.potassium as kmod
    bm.set_dt(0.1)

    class Net(bp.dyn.CondNeuGroup):
        def __init__(self, size):
            super().__init__(size)
            self.IKDR = kmod.IKDR_Ba2002(size)
            self.IK_TM = kmod.IK_TM1991(size)
            self.IK_HH = kmod.IK_HH1952(size)
            self.IKA1 = kmod.IKA1_HM1992(size)
            self.IKA2 = kmod.IKA2_HM1992(size)
            self.IKK2A = kmod.IKK2A_HM1992(size)
            self.IKK2B = kmod.IKK2B_HM1992(size)
            self.IKNI = kmod.IKNI_Ya1989(size)

    _run_cond_neu_group(Net(2))

    # The in-file legacy rate functions must also be NaN-free at their singular V.
    _assert_finite_with_grad(lambda: kmod.IK_HH1952(1), "f_p_alpha", -55.0)
    _assert_finite_with_grad(lambda: kmod.IKDR_Ba2002(1), "f_p_alpha", -35.0)
    _assert_finite_with_grad(lambda: kmod.IK_TM1991(1), "f_p_alpha", -45.0)


def test_coverage_calcium_fixed_channels():
    """Voltage-gated calcium channels hosted by a (fixed) Calcium ion."""
    bm.set_dt(0.1)

    class Net(bp.dyn.CondNeuGroupLTC):
        def __init__(self, size):
            super().__init__(size)
            self.Ca = bp.dyn.CalciumFixed(size)
            self.Ca.add_elem(
                icat_hm=bp.dyn.ICaT_HM1992(size),
                icat_hp=bp.dyn.ICaT_HP1992(size),
                icaht_hm=bp.dyn.ICaHT_HM1992(size),
                icaht_re=bp.dyn.ICaHT_Re1993(size),
                ical=bp.dyn.ICaL_IS2008(size),
            )

    _run_cond_neu_group(Net(2))


def test_coverage_calcium_dyna_channel_ican():
    """ICaN_IS2008 requires a CalciumDyna host; exercise it for coverage."""
    bm.set_dt(0.1)

    class Net(bp.dyn.CondNeuGroupLTC):
        def __init__(self, size):
            super().__init__(size)
            self.Ca = bp.dyn.CalciumDetailed(size)
            self.Ca.add_elem(ican=bp.dyn.ICaN_IS2008(size))

    _run_cond_neu_group(Net(2))


def test_coverage_ion_objects_reset_and_current():
    """Construct PotassiumFixed / CalciumFixed / SodiumFixed and exercise the ion API."""
    bm.set_dt(0.1)
    V = bm.ones(2) * -65.

    k = bp.dyn.PotassiumFixed(2, E=-95.)
    k.add_elem(ik=bp.dyn.IK_HH1952v2(2))
    na = bp.dyn.SodiumFixed(2, E=50.)
    na.add_elem(ina=bp.dyn.INa_HH1952v2(2))
    ca = bp.dyn.CalciumFixed(2)
    ca.add_elem(ical=bp.dyn.ICaL_IS2008(2))
    ca_dyn = bp.dyn.CalciumFirstOrder(2)

    for ion in (k, na, ca):
        ion.reset_state(V)
        bp.share.save(t=0., dt=0.1, i=0)
        ion.update(V)
        cur = np.asarray(ion.current(V))
        assert np.all(np.isfinite(cur))

    # CalciumDyna reset path (different reset_state signature).
    ca_dyn.reset_state(V)
    assert np.all(np.isfinite(np.asarray(ca_dyn.C)))
