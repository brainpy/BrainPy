# -*- coding: utf-8 -*-
"""Coverage tests for ``brainpy.integrators.ode.base`` (``ODEIntegrator``).

The existing ``delay_ode_test.py`` covers the basic state-delay path; this
file targets the remaining branches:

* the ``DT`` reserved-keyword guard (variable/parameter named ``dt``),
* the ``f_names`` non-identifier helper branch,
* ``neutral_delays`` construction + the read-only property setter error,
* the neutral-delay update path inside ``__call__`` for both ``NeuTimeDelay``
  and ``NeuLenDelay``,
* the state-delay update path for ``TimeDelay`` and ``LengthDelay``,
* the multi-variable ``dict_vars`` branch in ``__call__``.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import brainpy as bp
import brainpy.math as bm
from brainpy._errors import CodeError
from brainpy.integrators.ode import base
from brainpy.integrators.ode.base import ODEIntegrator, f_names


def f_simple(x, t):
    return -x


class TestConstructorGuards:
    def test_dt_reserved_keyword_as_param(self):
        # a parameter literally named ``dt`` is rejected
        def bad(x, t, dt):
            return -x

        with pytest.raises(CodeError):
            bp.odeint(bad, method='euler')

    def test_f_names_non_identifier(self):
        # f_names appends the function name only when it is a valid identifier;
        # a lambda has name "<lambda>" which is *not* an identifier, exercising
        # the false branch (the function's name is not appended).
        name = f_names(lambda x, t: -x)
        assert '<lambda>' not in name and 'lambda' not in name

    def test_f_names_identifier(self):
        name = f_names(f_simple)
        assert name.endswith('_f_simple')


class TestNeutralDelaysProperty:
    def test_setter_is_read_only(self):
        intg = bp.odeint(f_simple, method='euler')
        with pytest.raises(ValueError):
            intg.neutral_delays = {}

    def test_neutral_delays_property_default_empty(self):
        intg = bp.odeint(f_simple, method='euler')
        assert intg.neutral_delays == {}

    def test_neutral_delay_unknown_variable_raises(self):
        # a neutral-delay key that is not a declared variable raises DiffEqError
        nd = bm.NeuTimeDelay(bm.zeros(1), 1.0, dt=0.1)
        with pytest.raises(bp.errors.DiffEqError):
            bp.odeint(f_simple, method='euler', neutral_delays={'not_a_var': nd})


class TestNeutralDelayUpdate:
    def test_neu_time_delay_update_is_broken(self):
        # NOTE (defect): ODEIntegrator.__call__ updates a NeuTimeDelay via
        #   delay.update(kwargs['t'] + dt, new_devs[key])      (base.py ~line 145)
        # but ``NeuTimeDelay.update`` only accepts a single ``value`` argument,
        # so stepping an integrator that carries a NeuTimeDelay raises TypeError.
        bm.random.seed(0)
        nd = bm.NeuTimeDelay(bm.zeros(1), 1.0, dt=0.1)
        intg = bp.odeint(lambda x, t: -x, method='euler', dt=0.1,
                         neutral_delays={'x': nd})
        x = bm.zeros(1)
        with pytest.raises(TypeError):
            intg(x, 0.0)

    def test_neu_len_delay_rejected_at_construction(self):
        # NOTE (defect/limitation): the NeuLenDelay branch of the __call__
        # neutral-delay update (base.py ~line 143-144) is unreachable through
        # the public constructor: ``__init__`` validates neutral_delays with
        # ``is_dict_data(..., val_type=NeuTimeDelay)``, which rejects a
        # NeuLenDelay outright.
        nd = bm.NeuLenDelay(bm.zeros(1), 5)
        with pytest.raises(ValueError):
            bp.odeint(lambda x, t: -x, method='euler', dt=0.1,
                      neutral_delays={'x': nd})


class TestStateDelayUpdate:
    def test_length_delay_update_path(self):
        bm.random.seed(0)
        ld = bm.LengthDelay(bm.zeros(1), 5)
        intg = bp.odeint(lambda x, t: -x, method='euler', dt=0.1,
                         state_delays={'x': ld})
        x = bm.zeros(1)
        for i in range(5):
            x = intg(x, i * 0.1)
        assert np.all(np.isfinite(np.asarray(bm.as_jax(x))))

    def test_time_delay_update_path(self):
        bm.random.seed(0)
        td = bm.TimeDelay(bm.zeros(1), 1.0, before_t0=0.0, dt=0.1)
        intg = bp.odeint(lambda x, t: -x, method='euler', dt=0.1,
                         state_delays={'x': td})
        x = bm.zeros(1)
        for i in range(5):
            x = intg(x, i * 0.1)
        assert np.all(np.isfinite(np.asarray(bm.as_jax(x))))


class TestMultiVariable:
    def test_dict_vars_branch(self):
        # a JointEq integrator returns a list; the __call__ dict_vars branch
        # for len(variables) > 1 is taken.
        dx = lambda x, t, y: -x + y
        dy = lambda y, t, x: -y + x
        intg = bp.odeint(bp.JointEq(dx, dy), method='rk4', dt=0.1)
        out = intg(1.0, 0.5, 0.0)
        assert len(out) == 2
        assert all(np.all(np.isfinite(np.asarray(bm.as_jax(o)))) for o in out)


class TestModuleExports:
    def test_exports(self):
        assert hasattr(base, 'ODEIntegrator')
        assert ODEIntegrator.__name__ == 'ODEIntegrator'
