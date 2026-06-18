# -*- coding: utf-8 -*-
"""Coverage tests for ``brainpy/train/online.py``.

Exercises the online trainers :class:`brainpy.OnlineTrainer` and
:class:`brainpy.ForceTrainer` on a tiny echo-state network (ESN: a
``bp.dyn.Reservoir`` feeding a trainable ``bp.dnn.Dense`` readout).  Covered:

* ``fit_method`` resolution: ``None`` -> default RLS, a string shortcut, a dict
  spec, an explicit ``OnlineAlgorithm`` instance, and the not-callable error;
* training-node discovery error (no trainable node) and the ``_check_interface``
  ``TypeError`` for a trainable node that is not a :class:`SupportOnline`;
* :meth:`OnlineTrainer.fit` over an ``(X, Y)`` pair with ``reset_state``,
  ``data_first_axis`` 'B' / 'T', monitors (driving the ``_step_func_monitor``
  variable / callable / indexed branches), ``numpy_mon_after_run`` and the
  tqdm progress bar, plus the ``fit_record`` clearing in :meth:`predict`;
* the fit-data validation errors (non-(X, Y) container, wrong length);
* :class:`ForceTrainer` (FORCE learning, RLS-backed) construction + fit.
"""

import numpy as np
import pytest

import brainpy as bp
import brainpy.math as bm
from brainpy.algorithms.online import RLS, OnlineAlgorithm
from brainpy.mixin import SupportOnline


# ---------------------------------------------------------------------------
# Tiny echo-state network with a trainable Dense readout
# ---------------------------------------------------------------------------

class ESN(bp.DynamicalSystem):
    def __init__(self, num_in, num_hidden, num_out):
        super().__init__()
        self.r = bp.dyn.Reservoir(
            num_in, num_hidden,
            Win_initializer=bp.init.Uniform(-0.1, 0.1),
            Wrec_initializer=bp.init.Normal(scale=0.1),
            in_connectivity=0.1, rec_connectivity=0.1,
            comp_type='dense',
        )
        self.o = bp.dnn.Dense(num_hidden, num_out,
                              W_initializer=bp.init.Normal(),
                              mode=bm.training_mode)

    def update(self, x):
        return x >> self.r >> self.o


def _make_esn(num_in=3, num_hidden=10, num_out=2, seed=0):
    bm.random.seed(seed)
    bp.share.save(fit=True)
    with bm.batching_environment():
        return ESN(num_in, num_hidden, num_out)


def _xy(num_in=3, num_out=2, num_time=8, num_batch=1):
    x = bm.random.random((num_batch, num_time, num_in))
    y = bm.random.random((num_batch, num_time, num_out))
    return x, y


# NOTE on unreached lines (documented, not asserted):
#   * ``online.py:252`` -- the ``if shared_args is None`` guard inside
#     ``_step_func_fit`` is dead code: ``_fit`` always binds ``shared_args`` to a
#     ``DotDict`` via ``functools.partial`` before the for-loop, so it is never
#     ``None`` at that point.


# ---------------------------------------------------------------------------
# fit_method resolution
# ---------------------------------------------------------------------------

def test_default_fit_method_is_rls():
    """``fit_method=None`` -> a default RLS instance."""
    model = _make_esn()
    trainer = bp.OnlineTrainer(model, progress_bar=False)
    assert isinstance(trainer.fit_method, RLS)


def test_string_fit_method():
    """A string shortcut resolves through the online-method registry."""
    model = _make_esn()
    trainer = bp.OnlineTrainer(model, fit_method='rls', progress_bar=False)
    assert isinstance(trainer.fit_method, RLS)


def test_dict_fit_method():
    """A dict spec with ``name`` + init kwargs builds the algorithm."""
    model = _make_esn()
    trainer = bp.OnlineTrainer(model, fit_method={'name': 'rls', 'alpha': 0.1},
                               progress_bar=False)
    assert isinstance(trainer.fit_method, RLS)


def test_instance_fit_method():
    """An explicit ``OnlineAlgorithm`` instance is used as-is."""
    model = _make_esn()
    method = RLS(alpha=0.1)
    trainer = bp.OnlineTrainer(model, fit_method=method, progress_bar=False)
    assert trainer.fit_method is method


def test_not_callable_fit_method_raises():
    """A non-callable ``fit_method`` raises ``ValueError``."""
    model = _make_esn()

    class NotCallable:
        pass

    with pytest.raises(ValueError):
        bp.OnlineTrainer(model, fit_method=NotCallable(), progress_bar=False)


# ---------------------------------------------------------------------------
# Trainable-node discovery / interface checks
# ---------------------------------------------------------------------------

def test_no_trainable_nodes_raises():
    """A model with no ``TrainingMode`` node raises ``ValueError``."""
    class Plain(bp.DynamicalSystem):
        def __init__(self):
            super().__init__()
            self.n = bp.dyn.LifRef(3)

        def update(self, x):
            return self.n(x)

    with bm.batching_environment():
        plain = Plain()
    with pytest.raises(ValueError):
        bp.OnlineTrainer(plain, progress_bar=False)


def test_check_interface_rejects_non_support_online():
    """A trainable node that is not a ``SupportOnline`` raises ``TypeError``."""
    class NotOnline(bp.DynamicalSystem):
        def __init__(self):
            super().__init__()
            self.w = bm.TrainVar(bm.ones((2, 2)))

        def update(self, x):
            return bm.as_jax(x) @ self.w

    with bm.training_environment():
        node = NotOnline()
    assert isinstance(node.mode, bm.TrainingMode)
    assert not isinstance(node, SupportOnline)
    with pytest.raises(TypeError):
        bp.OnlineTrainer(node, progress_bar=False)


# ---------------------------------------------------------------------------
# fit() / predict()
# ---------------------------------------------------------------------------

def test_online_fit_and_predict_basic():
    """Basic fit() + predict() over an (X, Y) pair (batch-first)."""
    model = _make_esn()
    trainer = bp.OnlineTrainer(model, fit_method=RLS(alpha=0.1),
                               progress_bar=False)
    x, y = _xy()
    trainer.fit([x, y])
    out = trainer.predict(x)
    assert tuple(out.shape) == (1, 8, 2)
    # predict() clears each train node's fit_record
    for node in trainer.train_nodes:
        assert len(node.fit_record) == 0


def test_online_fit_with_reset_state_and_progress_bar():
    """fit() with ``reset_state=True`` and the tqdm progress bar."""
    model = _make_esn()
    trainer = bp.OnlineTrainer(model, fit_method=RLS(alpha=0.1),
                               progress_bar=True)
    x, y = _xy()
    trainer.fit([x, y], reset_state=True)
    assert trainer.i0 == 8


def test_online_fit_with_monitors_and_numpy_mon():
    """Monitors drive ``_step_func_monitor``; numpy_mon_after_run converts output."""
    model = _make_esn()
    trainer = bp.OnlineTrainer(model, fit_method=RLS(alpha=0.1),
                               monitors={'rstate': model.r.state},
                               numpy_mon_after_run=True, progress_bar=False)
    x, y = _xy()
    trainer.fit([x, y])
    assert isinstance(trainer.mon['rstate'], np.ndarray)
    # (num_batch, num_time, num_hidden)
    assert trainer.mon['rstate'].shape[:2] == (1, 8)


def test_online_fit_with_callable_monitor():
    """A callable monitor exercises the ``callable(val)`` branch of the monitor."""
    model = _make_esn()

    def mon_fun():
        # return a 1-D (per-batch) vector so the stacked history keeps a batch
        # axis for the trainer's ``moveaxis(x, 0, 1)`` post-processing.
        return bm.sum(model.r.state.value, axis=-1)

    trainer = bp.OnlineTrainer(model, fit_method=RLS(alpha=0.1),
                               monitors={'rsum': mon_fun}, progress_bar=False)
    x, y = _xy()
    trainer.fit([x, y])
    assert 'rsum' in trainer.mon


def test_online_fit_data_first_axis_time():
    """fit() with ``data_first_axis='T'`` -> data shaped (time, batch, feature)."""
    model = _make_esn()
    trainer = bp.OnlineTrainer(model, fit_method=RLS(alpha=0.1),
                               progress_bar=False, data_first_axis='T')
    # (num_time, num_batch, num_feature)
    x = bm.random.random((8, 1, 3))
    y = bm.random.random((8, 1, 2))
    trainer.fit([x, y])
    assert trainer.i0 == 8


# ---------------------------------------------------------------------------
# fit() data-validation errors
# ---------------------------------------------------------------------------

def test_online_fit_non_pair_container_raises():
    model = _make_esn()
    trainer = bp.OnlineTrainer(model, fit_method=RLS(alpha=0.1),
                               progress_bar=False)
    with pytest.raises(ValueError):
        trainer.fit({'x': 1})  # not a list/tuple


def test_online_fit_wrong_length_raises():
    model = _make_esn()
    trainer = bp.OnlineTrainer(model, fit_method=RLS(alpha=0.1),
                               progress_bar=False)
    with pytest.raises(ValueError):
        trainer.fit([1, 2, 3])  # length != 2


# ---------------------------------------------------------------------------
# Pinned defect: indexed monitor in OnlineTrainer (NOT in fix scope)
# ---------------------------------------------------------------------------

def test_indexed_monitor_in_online_fit_is_broken_defect():
    """PIN: an indexed monitor ``{'k': (var, idx)}`` breaks ``OnlineTrainer.fit``.

    ``online.py:294`` reads ``variable[bm.asarray(idx)]``.  Under the installed
    JAX, the ``bm.asarray(idx)`` index is a ``bm.Array`` whose
    ``__jax_array__`` is no longer honored during abstractification, so the
    indexed-monitor branch raises ``ValueError`` ("Triggering ``__jax_array__``
    during abstractification is no longer supported").  This is an API-drift
    defect in ``_step_func_monitor`` (not exercised by the other tests, which
    use whole-variable or callable monitors); pinned here so the regression is
    visible.  Plain-variable / callable monitors work (see the fit tests above).
    """
    model = _make_esn()
    trainer = bp.OnlineTrainer(
        model, fit_method=RLS(alpha=0.1),
        monitors={'rstate_col': (model.r.state, bm.asarray([0, 1]))},
        progress_bar=False)
    x, y = _xy()
    with pytest.raises(ValueError):
        trainer.fit([x, y])


# ---------------------------------------------------------------------------
# ForceTrainer
# ---------------------------------------------------------------------------

def test_force_trainer_fit():
    """ForceTrainer (FORCE learning) builds an RLS method and fits."""
    model = _make_esn()
    trainer = bp.ForceTrainer(model, alpha=0.1, progress_bar=False)
    assert isinstance(trainer.fit_method, RLS)
    x, y = _xy()
    trainer.fit([x, y])
    out = trainer.predict(x, reset_state=True)
    assert tuple(out.shape) == (1, 8, 2)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-q']))
