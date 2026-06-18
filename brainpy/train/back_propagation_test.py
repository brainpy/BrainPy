# -*- coding: utf-8 -*-
"""Coverage tests for ``brainpy/train/back_propagation.py``.

Exercises the back-propagation trainers :class:`brainpy.BPTT` (back-propagation
through time, recurrent) and :class:`brainpy.BPFF` (feedforward) end-to-end:

* trainer construction (default optimizer, string / callable / bad loss,
  deprecated ``seed`` / ``shuffle_data``, scalar-bool vs dict ``jit``, ``__repr__``);
* :meth:`BPTrainer.fit` over callable / iterable training data, with and without
  ``monitors`` (driving the ``predicts = (outs, mons)`` branch), ``loss_has_aux``,
  ``test_data``, ``num_report`` (per-step) vs ``num_report=-1`` (per-epoch),
  ``fun_after_report``, ``data_first_axis`` 'B' / 'T', and the tqdm progress bar;
* the history-metric accessors (``train_losses`` / ``test_losses`` /
  ``get_hist_metric`` for ``report`` and ``detailed``);
* :meth:`BPFF.predict` (including ``eval_time``);
* error branches: deprecated kwargs, unsupported loss type, ``(X, Y)`` tuple
  data, deprecated ``batch_size``, bad ``fun_after_report``, non-dict auxiliary
  data, and BPFF's ``data_first_axis != 'B'`` assertion.

.. note::
   The tiny models below deliberately implement ``update`` with raw
   :class:`brainpy.math.TrainVar` matmuls instead of ``bp.dnn.Dense`` /
   ``bp.dyn.RNNCell``.  Under the installed ``brainstate`` (0.5.x) the
   ``Dense.update`` line ``if share.load('fit', False) and self.online_fit_by``
   sees a *traced* ``fit`` flag inside the jitted/grad-traced fit loop and raises
   ``jax.errors.TracerBoolConversionError`` -- a real API-drift defect (see
   ``test_dense_layer_fit_flag_is_traced_defect``) that would otherwise block
   every BPTT/BPFF test.  Avoiding the ``fit``-flag read keeps these coverage
   tests independent of that defect.
"""

import numpy as np
import pytest

import brainpy as bp
import brainpy.math as bm
from brainpy._errors import NoLongerSupportError, UnsupportedError
from brainpy.train.back_propagation import BPTrainer, _is_brainpy_array


# ---------------------------------------------------------------------------
# Tiny models (no ``share.load('fit')`` read -> independent of the API drift)
# ---------------------------------------------------------------------------

class TinyRNN(bp.DynamicalSystem):
    """A minimal trainable recurrent network: h_t = tanh(x W_i + h_{t-1} W_h)."""

    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.n_hidden = n_hidden
        self.wi = bm.TrainVar(bm.random.random((n_in, n_hidden)) * 0.1)
        self.wh = bm.TrainVar(bm.random.random((n_hidden, n_hidden)) * 0.1)
        self.wo = bm.TrainVar(bm.random.random((n_hidden, n_out)) * 0.1)
        self.h = bm.Variable(bm.zeros((1, n_hidden)), batch_axis=0)

    def reset_state(self, batch_size=1, **kwargs):
        self.h.value = bm.zeros((batch_size, self.n_hidden))

    def update(self, x):
        x = bm.as_jax(x)
        self.h.value = bm.tanh(x @ self.wi + self.h.value @ self.wh)
        return self.h.value @ self.wo


class TinyFF(bp.DynamicalSystem):
    """A minimal trainable feedforward layer: y = x W + b."""

    def __init__(self, n_in, n_out):
        super().__init__()
        self.w = bm.TrainVar(bm.random.random((n_in, n_out)) * 0.1)
        self.b = bm.TrainVar(bm.zeros(n_out))

    def update(self, x):
        return bm.as_jax(x) @ self.w + self.b


def _mse(predicts, targets):
    return bp.losses.mean_squared_error(predicts, targets)


def _make_rnn(n_in=2, n_hidden=4, n_out=1, seed=0):
    bm.random.seed(seed)
    with bm.training_environment():
        return TinyRNN(n_in, n_hidden, n_out)


def _make_ff(n_in=3, n_out=2, seed=0):
    bm.random.seed(seed)
    with bm.training_environment():
        return TinyFF(n_in, n_out)


# ---------------------------------------------------------------------------
# Constructor coverage
# ---------------------------------------------------------------------------

def test_bptt_default_optimizer_and_repr():
    """Default optimizer (Adam + ExponentialDecay) is built and registered."""
    model = _make_rnn()
    trainer = bp.BPTT(model, loss_fun=_mse, progress_bar=False)
    assert isinstance(trainer.optimizer, bp.optim.Adam)
    # train vars were auto-registered with the optimizer
    assert len(trainer.optimizer.vars_to_train) > 0
    text = repr(trainer)
    assert 'BPTT' in text and 'optimizer' in text


def test_bptt_string_loss_function():
    """A string ``loss_fun`` resolves to a function in ``brainpy.losses``."""
    model = _make_rnn()
    trainer = bp.BPTT(model, loss_fun='mean_squared_error', progress_bar=False)
    assert trainer._loss_func is bp.losses.mean_squared_error


def test_bptt_bad_loss_type_raises():
    """A non-str / non-callable loss raises ``UnsupportedError``."""
    model = _make_rnn()
    with pytest.raises(UnsupportedError):
        bp.BPTT(model, loss_fun=123, progress_bar=False)


def test_bptt_deprecated_kwargs_raise():
    """Deprecated ``shuffle_data`` / ``seed`` kwargs raise ``NoLongerSupportError``."""
    model = _make_rnn()
    with pytest.raises(NoLongerSupportError):
        bp.BPTT(model, loss_fun=_mse, shuffle_data=True, progress_bar=False)
    with pytest.raises(NoLongerSupportError):
        bp.BPTT(model, loss_fun=_mse, seed=1, progress_bar=False)


def test_bptt_dict_jit_config():
    """A dict ``jit`` configures the per-phase jit flags."""
    model = _make_rnn()
    trainer = bp.BPTT(model, loss_fun=_mse,
                      jit={'predict': True, 'fit': False, 'loss': True},
                      progress_bar=False)
    from brainpy.running import constants as c
    assert trainer.jit[c.FIT_PHASE] is False
    assert trainer.jit[c.PREDICT_PHASE] is True
    assert trainer.jit[c.LOSS_PHASE] is True


# ---------------------------------------------------------------------------
# BPTT.fit coverage
# ---------------------------------------------------------------------------

def test_bptt_fit_callable_data_per_epoch_report():
    """fit() with a callable dataset, default ``num_report=-1`` (per-epoch)."""
    model = _make_rnn()
    trainer = bp.BPTT(model, loss_fun=_mse, optimizer=bp.optim.Adam(lr=0.02),
                      progress_bar=False)
    X = bm.random.random((3, 5, 2))
    Y = bm.random.random((3, 5, 1))

    def train_data():
        yield X, Y

    trainer.fit(train_data, num_epoch=2)
    losses = trainer.train_losses
    assert losses is not None and len(losses) == 2
    # detailed metrics accumulate per-step losses too
    detailed = trainer.get_hist_metric(phase='fit', which='detailed')
    assert detailed is not None and len(detailed) == 2


def test_bptt_fit_iterable_with_progress_bar():
    """fit() with an iterable (list) dataset drives the tqdm progress-bar path."""
    model = _make_rnn()
    trainer = bp.BPTT(model, loss_fun=_mse, optimizer=bp.optim.Adam(lr=0.02),
                      progress_bar=True)
    X = bm.random.random((3, 5, 2))
    Y = bm.random.random((3, 5, 1))
    trainer.fit([(X, Y)], num_epoch=1)
    assert trainer.train_losses is not None


def test_bptt_fit_with_monitors_aux_report_and_test_data():
    """Covers monitors (``(outs, mons)`` branch), aux loss, num_report>0, test set."""
    model = _make_rnn()

    def loss_aux(predicts, targets):
        # with monitors set, ``predicts`` is the (outputs, monitors) tuple
        outs, mons = predicts
        l = bp.losses.mean_squared_error(outs, targets)
        return l, {'mse': l}

    trainer = bp.BPTT(model, loss_fun=loss_aux, optimizer=bp.optim.Adam(lr=0.02),
                      loss_has_aux=True, monitors={'h': model.h},
                      progress_bar=False)
    X = bm.random.random((3, 5, 2))
    Y = bm.random.random((3, 5, 1))

    reports = []

    def after(idx, metrics, phase):
        reports.append((phase, sorted(metrics.keys())))

    def gen():
        yield X, Y
        yield X, Y

    trainer.fit(gen, test_data=gen, num_epoch=1, num_report=1,
                fun_after_report=after)

    # aux metric "mse" recorded for both train and test
    assert trainer.get_hist_metric(phase='fit', metric='mse') is not None
    assert trainer.test_losses is not None
    phases = {p for p, _ in reports}
    assert phases == {'fit', 'test'}
    assert all('mse' in keys for _, keys in reports)


def test_bptt_fit_data_first_axis_time():
    """fit() with ``data_first_axis='T'`` -> data shaped (time, batch, feature)."""
    model = _make_rnn()
    trainer = bp.BPTT(model, loss_fun=_mse, optimizer=bp.optim.Adam(lr=0.02),
                      progress_bar=False, data_first_axis='T')
    Xt = bm.random.random((5, 3, 2))
    Yt = bm.random.random((5, 3, 1))
    trainer.fit([(Xt, Yt)], num_epoch=1)
    assert trainer.train_losses is not None


def test_bptt_fit_num_report_with_progress_bar_and_test_data():
    """num_report>0 + progress_bar + test_data -> per-step report on both phases.

    Covers the ``bar.set_description`` branches in both the fit and test
    per-step report blocks.
    """
    model = _make_rnn()
    trainer = bp.BPTT(model, loss_fun=_mse, optimizer=bp.optim.Adam(lr=0.02),
                      progress_bar=True)
    X = bm.random.random((3, 5, 2))
    Y = bm.random.random((3, 5, 1))

    phases = []

    def after(idx, metrics, phase):
        phases.append(phase)

    def gen():
        yield X, Y
        yield X, Y

    trainer.fit(gen, test_data=gen, num_epoch=1, num_report=1,
                fun_after_report=after)
    assert set(phases) == {'fit', 'test'}
    assert trainer.test_losses is not None


def test_bptt_fit_per_epoch_report_with_progress_bar_and_test_data():
    """num_report=-1 + progress_bar + test_data -> per-epoch report on both phases.

    Covers the per-epoch ``bar.set_description`` + ``fun_after_report`` branches
    for the fit phase and the entire per-epoch test-report block.
    """
    model = _make_rnn()
    trainer = bp.BPTT(model, loss_fun=_mse, optimizer=bp.optim.Adam(lr=0.02),
                      progress_bar=True)
    X = bm.random.random((3, 5, 2))
    Y = bm.random.random((3, 5, 1))

    phases = []

    def after(idx, metrics, phase):
        phases.append(phase)

    trainer.fit([(X, Y)], test_data=[(X, Y)], num_epoch=1, num_report=-1,
                fun_after_report=after)
    assert phases == ['fit', 'test']
    assert trainer.train_losses is not None and trainer.test_losses is not None


def test_bptt_fit_num_report_bar_with_list_datasets():
    """num_report>0 with *list* (sized) train+test data -> tqdm ``set_description``.

    A list has ``__len__`` so the trainer builds a tqdm ``bar`` and reports via
    ``bar.set_description`` (rather than ``print``) in both the fit and test
    per-step report blocks.
    """
    model = _make_rnn()
    trainer = bp.BPTT(model, loss_fun=_mse, optimizer=bp.optim.Adam(lr=0.02),
                      progress_bar=True)
    X = bm.random.random((3, 5, 2))
    Y = bm.random.random((3, 5, 1))
    # a 3-element list avoids the ``len(train_data) == 2`` (bare X,Y) guard while
    # still exposing ``__len__`` so the trainer builds a tqdm progress bar.
    train = [(X, Y), (X, Y), (X, Y)]
    test = [(X, Y), (X, Y), (X, Y)]
    trainer.fit(train, test_data=test, num_epoch=1, num_report=1)
    assert trainer.train_losses is not None
    assert trainer.test_losses is not None


def test_bptt_fit_per_epoch_print_branch_with_generators():
    """num_report=-1 with un-sized (generator) data -> the ``print`` report path.

    Generators have no ``__len__`` so ``bar`` stays ``None`` and the per-epoch
    fit/test reports go through ``print`` instead of ``bar.set_description``.
    """
    model = _make_rnn()
    trainer = bp.BPTT(model, loss_fun=_mse, optimizer=bp.optim.Adam(lr=0.02),
                      progress_bar=False)
    X = bm.random.random((3, 5, 2))
    Y = bm.random.random((3, 5, 1))

    def gen():
        yield X, Y

    trainer.fit(gen, test_data=gen, num_epoch=1, num_report=-1)
    assert trainer.train_losses is not None
    assert trainer.test_losses is not None


# NOTE: ``back_propagation.py:384`` (the *test*-phase "aux is not a dict"
# TypeError) is unreachable through the public API: the trainer uses a single
# ``loss_fun`` for both the fit and test phases, so a non-dict aux always raises
# first in the fit phase (line 305/306, covered by
# ``test_bptt_fit_aux_not_dict_raises_typeerror``).  Line 268 (a second,
# redundant ``if shared_args is None`` block) is likewise dead code -- by that
# point ``shared_args`` has already been coerced to a ``DotDict``.


# ---------------------------------------------------------------------------
# BPTT.fit error branches
# ---------------------------------------------------------------------------

def test_bptt_fit_deprecated_batch_size_raises():
    model = _make_rnn()
    trainer = bp.BPTT(model, loss_fun=_mse, optimizer=bp.optim.Adam(lr=0.02),
                      progress_bar=False)
    with pytest.raises(NoLongerSupportError):
        trainer.fit([(bm.ones((2, 5, 2)), bm.ones((2, 5, 1)))], batch_size=4)


def test_bptt_fit_xy_tuple_raises_unsupported():
    """Passing a bare ``(X, Y)`` 2-tuple (not an iterable of pairs) is rejected."""
    model = _make_rnn()
    trainer = bp.BPTT(model, loss_fun=_mse, optimizer=bp.optim.Adam(lr=0.02),
                      progress_bar=False)
    with pytest.raises(UnsupportedError):
        trainer.fit((bm.ones((2, 5, 2)), bm.ones((2, 5, 1))))


def test_bptt_fit_bad_fun_after_report_raises():
    model = _make_rnn()
    trainer = bp.BPTT(model, loss_fun=_mse, optimizer=bp.optim.Adam(lr=0.02),
                      progress_bar=False)
    with pytest.raises(AssertionError):
        trainer.fit([(bm.ones((2, 5, 2)), bm.ones((2, 5, 1)))],
                    fun_after_report=123)


def test_bptt_fit_aux_not_dict_raises_typeerror():
    """``loss_has_aux=True`` but aux is not a dict -> ``TypeError``."""
    model = _make_rnn()

    def bad_aux(predicts, targets):
        l = bp.losses.mean_squared_error(predicts, targets)
        return l, [l]  # a list, not a dict

    trainer = bp.BPTT(model, loss_fun=bad_aux, optimizer=bp.optim.Adam(lr=0.01),
                      loss_has_aux=True, progress_bar=False)
    with pytest.raises(TypeError):
        trainer.fit([(bm.random.random((2, 4, 2)), bm.random.random((2, 4, 1)))],
                    num_epoch=1)


# ---------------------------------------------------------------------------
# get_hist_metric accessor coverage
# ---------------------------------------------------------------------------

def test_get_hist_metric_unset_returns_none():
    """Accessors return ``None`` for metrics/phases that were never recorded."""
    model = _make_rnn()
    trainer = bp.BPTT(model, loss_fun=_mse, progress_bar=False)
    assert trainer.get_hist_metric(phase='train') is None
    assert trainer.get_hist_metric(phase='predict') is None
    assert trainer.get_hist_metric(phase='fit', which='detailed') is None
    assert trainer.get_hist_metric(phase='test', which='detailed') is None
    assert trainer.train_losses is None
    assert trainer.test_losses is None


# ---------------------------------------------------------------------------
# BPFF coverage
# ---------------------------------------------------------------------------

def test_bpff_fit_and_predict():
    """BPFF fit() (feedforward, no time axis) + predict() + eval_time."""
    model = _make_ff()
    trainer = bp.BPFF(model, loss_fun=_mse, optimizer=bp.optim.Adam(lr=0.05),
                      progress_bar=False)
    X = bm.random.random((4, 3))
    Y = bm.random.random((4, 2))

    def train_data():
        yield X, Y

    trainer.fit(train_data, num_epoch=2)
    assert trainer.train_losses is not None and len(trainer.train_losses) == 2

    out = trainer.predict(X)
    assert tuple(out.shape) == (4, 2)

    # eval_time returns (elapsed, output)
    elapsed, out2 = trainer.predict(X, eval_time=True)
    assert elapsed >= 0.0
    assert tuple(out2.shape) == (4, 2)


def test_bpff_predict_with_monitors_and_numpy_mon():
    """BPFF predict() populates monitors and converts them to numpy arrays."""
    model = _make_ff()
    trainer = bp.BPFF(model, loss_fun=_mse, optimizer=bp.optim.Adam(lr=0.05),
                      monitors={'w': model.w}, numpy_mon_after_run=True,
                      progress_bar=False)
    out = trainer.predict(bm.random.random((4, 3)))
    assert tuple(out.shape) == (4, 2)
    assert isinstance(trainer.mon['w'], np.ndarray)


def test_bpff_fit_with_monitors_uses_mon_branch():
    """With monitors set, BPFF passes ``(outputs, mon)`` to the loss function."""
    model = _make_ff()

    def loss_aux(predicts, targets):
        outs, mon = predicts
        return bp.losses.mean_squared_error(outs, targets)

    trainer = bp.BPFF(model, loss_fun=loss_aux, optimizer=bp.optim.Adam(lr=0.05),
                      monitors={'w': model.w}, progress_bar=False)
    trainer.fit([(bm.random.random((4, 3)), bm.random.random((4, 2)))],
                num_epoch=1)
    assert trainer.train_losses is not None


def test_bpff_predict_without_jit():
    """BPFF predict() with jit disabled -> the non-jit ``_fun_predict`` branch.

    .. note::
       The ``jit={'predict': False}`` *constructor* kwarg is silently ignored:
       ``brainpy/running/runner.py:101`` does ``jit.pop('predict', True)`` which
       mutates the dict, so by the time ``BPTrainer.__init__`` re-reads
       ``self._origin_jit`` the ``'predict'`` key is gone and ``jit['predict']``
       defaults back to ``True``.  We therefore flip ``trainer.jit['predict']``
       directly to drive the non-jit ``_fun_predict`` else-branch.
    """
    model = _make_ff()
    trainer = bp.BPFF(model, loss_fun=_mse, optimizer=bp.optim.Adam(lr=0.05),
                      progress_bar=False)
    assert trainer.jit['predict'] is True  # constructor kwarg was ignored
    trainer.jit['predict'] = False
    out = trainer.predict(bm.random.random((4, 3)))
    assert tuple(out.shape) == (4, 2)


def test_bpff_data_first_axis_time_asserts():
    """BPFF requires ``data_first_axis='B'``; 'T' triggers the assertion."""
    model = _make_ff()
    trainer = bp.BPFF(model, loss_fun=_mse, optimizer=bp.optim.Adam(lr=0.05),
                      progress_bar=False, data_first_axis='T')
    with pytest.raises(AssertionError):
        trainer.fit([(bm.random.random((4, 3)), bm.random.random((4, 2)))],
                    num_epoch=1)


# ---------------------------------------------------------------------------
# Misc helpers / abstract base
# ---------------------------------------------------------------------------

def test_is_brainpy_array_helper():
    """The module-level ``_is_brainpy_array`` type guard."""
    assert _is_brainpy_array(bm.Array(bm.ones(2))) is True
    assert _is_brainpy_array(5) is False
    assert _is_brainpy_array(bm.as_jax(bm.ones(2))) is False


def test_bptrainer_abstract_step_funcs_raise():
    """``BPTrainer`` is abstract: its step functions raise ``NotImplementedError``."""
    model = _make_ff()
    trainer = BPTrainer(model, loss_fun=_mse, progress_bar=False)
    with pytest.raises(NotImplementedError):
        BPTrainer._step_func_loss(trainer, {}, bm.ones((2, 3)), bm.ones((2, 2)))
    with pytest.raises(NotImplementedError):
        BPTrainer._step_func_fit(trainer, {}, bm.ones((2, 3)), bm.ones((2, 2)))


# ---------------------------------------------------------------------------
# Pinned defect (NOT in fix scope -- documents current behavior)
# ---------------------------------------------------------------------------

def test_dense_layer_fit_flag_is_traced_defect():
    """PIN: ``bp.dnn.Dense`` under a BPFF/BPTT fit loop raises on the ``fit`` flag.

    ``brainpy/dnn/linear.py:129`` does
    ``if share.load('fit', False) and self.online_fit_by is not None:``.
    Under the installed ``brainstate`` (0.5.x), inside the jitted / grad-traced
    fit step the ``fit`` flag is a JAX *tracer*, so the boolean ``and`` raises
    ``jax.errors.TracerBoolConversionError``.  This blocks the canonical
    ``RNNCell``/``Dense`` BPTT example.  It is an API-drift defect in the layer,
    not in ``back_propagation.py``; pinned here so the regression is visible.
    """
    import jax

    class DenseFF(bp.DynamicalSystem):
        def __init__(self):
            super().__init__()
            self.lin = bp.dnn.Dense(3, 2, mode=bm.training_mode)

        def update(self, x):
            return self.lin(x)

        def reset_state(self, batch_size=1, **kwargs):
            pass

    with bm.training_environment():
        model = DenseFF()
    trainer = bp.BPFF(model, loss_fun=_mse, optimizer=bp.optim.Adam(lr=0.01),
                      progress_bar=False)
    with pytest.raises(jax.errors.TracerBoolConversionError):
        trainer.fit([(bm.random.random((4, 3)), bm.random.random((4, 2)))],
                    num_epoch=1)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-q']))
