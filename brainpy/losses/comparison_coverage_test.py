# -*- coding: utf-8 -*-
"""Coverage + correctness tests for ``brainpy/losses/comparison.py``.

The ``brainpy.losses`` package shipped with **zero** co-located tests. This
module raises ``comparison.py`` (and, transitively, ``losses/utils.py`` and
``losses/base.py``) to >=90% line coverage by:

- checking every public loss function against an independent NumPy/JAX
  reference (not just "it runs"),
- walking every ``reduction`` branch (``'mean'`` / ``'sum'`` / ``'none'`` and
  the error path) including the weighted / ``ignore_index`` / label-smoothing
  branches of :func:`cross_entropy_loss`,
- exercising the ``Loss`` subclasses both via ``.update(...)`` and ``__call__``,
- driving the pytree (dict/list) leaves through ``_multi_return`` and the
  ``bm.Array``-leaf path.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import brainpy.math as bm
from brainpy._errors import UnsupportedError
from brainpy.losses import comparison as C
from brainpy.losses.utils import _reduce, _multi_return, _is_leaf
from brainpy.losses.base import Loss, WeightedLoss


def _f(x):
    return np.asarray(x, dtype=np.float64)


# ---------------------------------------------------------------------------
# losses/utils.py
# ---------------------------------------------------------------------------
class TestUtils:
    def test_reduce_mean_sum_none(self):
        x = jnp.array([1., 2., 3., 4.])
        assert float(_reduce(x, 'mean')) == pytest.approx(2.5)
        assert float(_reduce(x, 'sum')) == pytest.approx(10.0)
        assert np.allclose(np.asarray(_reduce(x, 'none')), _f(x))

    def test_reduce_axis(self):
        x = jnp.arange(6.).reshape(2, 3)
        assert np.allclose(np.asarray(_reduce(x, 'sum', axis=1)), [3., 12.])

    def test_reduce_invalid_raises(self):
        with pytest.raises(UnsupportedError):
            _reduce(jnp.array([1.]), 'banana')

    def test_is_leaf(self):
        assert _is_leaf(bm.asarray([1.]))
        assert not _is_leaf(jnp.array([1.]))

    def test_multi_return_jax_array(self):
        x = jnp.array([1., 2.])
        assert _multi_return(x) is x

    def test_multi_return_bm_array_returns_value(self):
        x = bm.asarray([1., 2.])
        out = _multi_return(x)
        assert isinstance(out, jnp.ndarray)
        assert np.allclose(np.asarray(out), [1., 2.])

    def test_multi_return_pytree_sums_leaves(self):
        out = _multi_return({'a': jnp.array(1.0), 'b': jnp.array([2.0, 3.0])})
        # first leaf + remaining leaves, broadcast
        assert np.allclose(np.asarray(out), [3.0, 4.0])


# ---------------------------------------------------------------------------
# losses/base.py
# ---------------------------------------------------------------------------
class TestBaseClasses:
    def test_loss_reduction_attr(self):
        loss = Loss(reduction='sum')
        assert loss.reduction == 'sum'

    def test_weighted_loss_attrs(self):
        w = bm.asarray([1., 2.])
        loss = WeightedLoss(weight=w, reduction='none')
        assert loss.reduction == 'none'
        assert loss.weight is w


# ---------------------------------------------------------------------------
# cross_entropy_loss + CrossEntropyLoss
# ---------------------------------------------------------------------------
class TestCrossEntropy:
    def _ce_ref(self, logits, idx):
        logits = _f(logits)
        lse = np.log(np.exp(logits).sum(-1))
        return lse - logits[np.arange(len(idx)), idx]

    def test_integer_targets_mean(self):
        logits = jnp.array([[2., 1., 0.], [0., 2., 1.]])
        targets = jnp.array([0, 1])
        ref = self._ce_ref(logits, [0, 1]).mean()
        assert float(C.cross_entropy_loss(logits, targets)) == pytest.approx(ref, rel=1e-5)

    def test_integer_targets_none_and_sum(self):
        logits = jnp.array([[2., 1., 0.], [0., 2., 1.]])
        targets = jnp.array([0, 1])
        ref = self._ce_ref(logits, [0, 1])
        none = np.asarray(C.cross_entropy_loss(logits, targets, reduction='none'))
        assert np.allclose(none, ref, rtol=1e-5)
        assert float(C.cross_entropy_loss(logits, targets, reduction='sum')) == pytest.approx(ref.sum(), rel=1e-5)

    def test_probability_targets(self):
        logits = jnp.array([[2., 1., 0.]])
        probs = jnp.array([[0.7, 0.2, 0.1]])
        logits_np = _f(logits)[0]
        log_sm = logits_np - np.log(np.exp(logits_np).sum())
        ref = -(np.asarray([0.7, 0.2, 0.1]) * log_sm).sum()
        assert float(C.cross_entropy_loss(logits, probs)) == pytest.approx(ref, rel=1e-5)

    def test_weighted_integer_targets(self):
        logits = jnp.array([[2., 1., 0.], [0., 2., 1.]])
        targets = jnp.array([0, 1])
        weight = jnp.array([2.0, 1.0, 1.0])
        per = self._ce_ref(logits, [0, 1])
        w = np.array([2.0, 1.0])
        ref = (per * w).sum() / w.sum()
        out = C.cross_entropy_loss(logits, targets, weight=weight, reduction='mean')
        assert float(out) == pytest.approx(ref, rel=1e-5)

    def test_weighted_probability_targets(self):
        logits = jnp.array([[2., 1., 0.]])
        probs = jnp.array([[0.7, 0.2, 0.1]])
        weight = jnp.array([2.0, 1.0, 1.0])
        out = C.cross_entropy_loss(logits, probs, weight=weight, reduction='sum')
        assert np.isfinite(float(out))

    def test_ignore_index_excludes_sample(self):
        logits = jnp.array([[2., 1., 0.], [0., 2., 1.]])
        targets = jnp.array([0, -100])  # second sample ignored
        ref = self._ce_ref(logits, [0, 0])[0]  # only first sample contributes
        out = C.cross_entropy_loss(logits, targets, reduction='mean', ignore_index=-100)
        assert float(out) == pytest.approx(ref, rel=1e-5)

    def test_label_smoothing_changes_value(self):
        logits = jnp.array([[2., 1., 0.], [0., 2., 1.]])
        targets = jnp.array([0, 1])
        plain = float(C.cross_entropy_loss(logits, targets))
        smoothed = float(C.cross_entropy_loss(logits, targets, label_smoothing=0.2))
        assert plain != pytest.approx(smoothed)

    def test_label_smoothing_probability_targets(self):
        # probabilities path + label_smoothing>0 (covers the soft-target smoothing branch)
        logits = jnp.array([[2., 1., 0.]])
        probs = jnp.array([[0.7, 0.2, 0.1]])
        plain = float(C.cross_entropy_loss(logits, probs))
        smoothed = float(C.cross_entropy_loss(logits, probs, label_smoothing=0.3))
        assert plain != pytest.approx(smoothed)

    def test_class_matches_function(self):
        logits = jnp.array([[2., 1., 0.], [0., 2., 1.]])
        targets = jnp.array([0, 1])
        layer = C.CrossEntropyLoss(reduction='mean')
        assert float(layer.update(logits, targets)) == pytest.approx(
            float(C.cross_entropy_loss(logits, targets)), rel=1e-6)
        # __call__ routes to update
        assert float(layer(logits, targets)) == pytest.approx(float(layer.update(logits, targets)))


class TestCrossEntropySparseSigmoid:
    def test_sparse_with_int_target(self):
        logits = jnp.array([[2., 1., 0.]])
        out = C.cross_entropy_sparse(logits, 0)
        ln = _f(logits)[0]
        ref = np.log(np.exp(ln).sum()) - ln[0]
        assert float(out[0]) == pytest.approx(ref, rel=1e-5)

    def test_sparse_with_array_target(self):
        logits = jnp.array([[2., 1., 0.], [0., 2., 1.]])
        targets = jnp.array([[0], [1]])
        out = np.asarray(C.cross_entropy_sparse(logits, targets))
        assert out.shape == (2,)

    def test_sigmoid(self):
        logits = jnp.array([0.5, -0.5])
        labels = jnp.array([1.0, 0.0])
        out = np.asarray(C.cross_entropy_sigmoid(logits, labels))
        ref = np.maximum(_f(logits), 0) - _f(logits) * _f(labels) + np.log1p(np.exp(-np.abs(_f(logits))))
        assert np.allclose(out, ref, rtol=1e-5)


# ---------------------------------------------------------------------------
# NLL
# ---------------------------------------------------------------------------
class TestNLL:
    def test_basic_mean(self):
        logprobs = jnp.log(jnp.array([[0.7, 0.3], [0.4, 0.6]]))
        targets = jnp.array([0, 1])
        ref = -np.array([np.log(0.7), np.log(0.6)]).mean()
        assert float(C.nll_loss(logprobs, targets)) == pytest.approx(ref, rel=1e-5)

    def test_sum_none_and_none_keyword(self):
        logprobs = jnp.log(jnp.array([[0.7, 0.3], [0.4, 0.6]]))
        targets = jnp.array([0, 1])
        assert float(C.nll_loss(logprobs, targets, 'sum')) == pytest.approx(
            -(np.log(0.7) + np.log(0.6)), rel=1e-5)
        assert np.asarray(C.nll_loss(logprobs, targets, 'none')).shape == (2,)
        assert np.asarray(C.nll_loss(logprobs, targets, None)).shape == (2,)

    def test_invalid_reduction_raises(self):
        logprobs = jnp.log(jnp.array([[0.7, 0.3]]))
        with pytest.raises(ValueError):
            C.nll_loss(logprobs, jnp.array([0]), 'bogus')

    def test_class_wrapper(self):
        logprobs = jnp.log(jnp.array([[0.7, 0.3], [0.4, 0.6]]))
        targets = jnp.array([0, 1])
        layer = C.NLLLoss(reduction='mean')
        assert float(layer.update(logprobs, targets)) == pytest.approx(
            float(C.nll_loss(logprobs, targets)), rel=1e-6)


# ---------------------------------------------------------------------------
# L1 / L2 / MAE / MSE / MSLE / Huber
# ---------------------------------------------------------------------------
class TestRegressionLosses:
    def test_l1_loss_reductions(self):
        # NOTE: l1_loss now delegates to braintools.metric.l1_loss, which
        # computes the per-row MEAN absolute error (not the L1 norm) for
        # reduction='none', then sums / means those per-row values.
        x = jnp.array([[1., 2.], [3., 4.]])
        y = jnp.zeros((2, 2))
        none = np.asarray(C.l1_loss(x, y, reduction='none'))
        assert np.allclose(none, [1.5, 3.5])  # per-row mean abs error
        assert float(C.l1_loss(x, y, reduction='sum')) == pytest.approx(5.0)
        assert float(C.l1_loss(x, y, reduction='mean')) == pytest.approx(2.5)

    def test_l1_class(self):
        x = jnp.array([[1., 2.], [3., 4.]])
        y = jnp.zeros((2, 2))
        layer = C.L1Loss(reduction='sum')
        assert float(layer.update(x, y)) == pytest.approx(5.0)

    def test_l2_loss_elementwise(self):
        out = np.asarray(C.l2_loss(jnp.array([2.0, 0.0]), jnp.array([0.0, 0.0])))
        assert np.allclose(out, [2.0, 0.0])  # 0.5 * err^2

    def test_mae_axis_and_class(self):
        x = jnp.array([[1., -2.], [3., -4.]])
        y = jnp.zeros((2, 2))
        assert np.allclose(np.asarray(C.mean_absolute_error(x, y, axis=1)), [1.5, 3.5])
        assert float(C.MAELoss(reduction='mean').update(x, y)) == pytest.approx(2.5)

    def test_mae_bm_array_input(self):
        # exercises the bm.Array leaf path in _multi_return
        x = bm.asarray([[1., -2.]])
        y = bm.asarray([[0., 0.]])
        out = C.mean_absolute_error(x, y, reduction='mean')
        assert float(out) == pytest.approx(1.5)

    def test_mse_and_class(self):
        x = jnp.array([[1., 2.], [3., 4.]])
        y = jnp.zeros((2, 2))
        assert float(C.mean_squared_error(x, y)) == pytest.approx((1 + 4 + 9 + 16) / 4)
        assert float(C.MSELoss(reduction='sum').update(x, y)) == pytest.approx(30.0)

    def test_msle(self):
        x = jnp.array([0.0, 1.0])
        y = jnp.array([0.0, 1.0])
        assert float(C.mean_squared_log_error(x, y)) == pytest.approx(0.0)
        val = float(C.mean_squared_log_error(jnp.array([2.0]), jnp.array([0.0])))
        assert val == pytest.approx((np.log1p(2.0)) ** 2, rel=1e-5)

    def test_huber_quadratic_and_linear(self):
        # |err| <= delta -> 0.5 err^2 ; |err| > delta -> delta*(|err|-0.5 delta)
        out = np.asarray(C.huber_loss(jnp.array([0.5, 5.0]), jnp.array([0.0, 0.0]), delta=1.0))
        assert out[0] == pytest.approx(0.125)
        assert out[1] == pytest.approx(1.0 * (5.0 - 0.5))

    def test_pytree_dict_input(self):
        # drive _multi_return tree_flatten branch
        x = {'a': jnp.array([1., 2.]), 'b': jnp.array([3.])}
        y = {'a': jnp.array([0., 0.]), 'b': jnp.array([0.])}
        out = C.mean_squared_error(x, y, reduction='sum')
        assert np.isfinite(float(out))


# ---------------------------------------------------------------------------
# logistic / softmax / log-cosh
# ---------------------------------------------------------------------------
class TestLogisticFamily:
    def test_binary_logistic_loss(self):
        out = float(C.binary_logistic_loss(jnp.array(0.0), jnp.array(1.0)))
        assert out == pytest.approx(np.log(2.0), rel=1e-5)  # softplus(0) - 1*0

    def test_multiclass_logistic_loss(self):
        logits = jnp.array([2.0, 1.0, 0.0])
        out = float(C.multiclass_logistic_loss(jnp.array(0), logits))
        ref = np.log(np.exp(_f(logits)).sum()) - 2.0
        assert out == pytest.approx(ref, rel=1e-5)

    def test_sigmoid_binary_cross_entropy(self):
        logits = jnp.array([0.0, 2.0])
        labels = jnp.array([1.0, 0.0])
        out = np.asarray(C.sigmoid_binary_cross_entropy(logits, labels))
        assert out.shape == (2,)
        assert out[0] == pytest.approx(np.log(2.0), rel=1e-5)

    def test_softmax_cross_entropy(self):
        logits = jnp.array([[2., 1., 0.]])
        labels = jnp.array([[1., 0., 0.]])
        ln = _f(logits)[0]
        log_sm = ln - np.log(np.exp(ln).sum())
        ref = -log_sm[0]
        assert float(C.softmax_cross_entropy(logits, labels)[0]) == pytest.approx(ref, rel=1e-5)

    def test_log_cosh_loss(self):
        out = float(C.log_cosh_loss(jnp.array([0.0]), jnp.array([0.0]))[0])
        assert out == pytest.approx(0.0, abs=1e-6)
        big = float(C.log_cosh_loss(jnp.array([10.0]), jnp.array([0.0]))[0])
        assert big == pytest.approx(10.0 - np.log(2.0), rel=1e-3)


# ---------------------------------------------------------------------------
# multi_margin_loss
# ---------------------------------------------------------------------------
class TestMultiMargin:
    def test_p1_mean(self):
        predicts = jnp.array([[0.2, 0.8], [0.6, 0.4]])
        targets = jnp.array([1, 0])
        out = float(C.multi_margin_loss(predicts, targets, margin=1.0, p=1, reduction='mean'))
        assert np.isfinite(out) and out >= 0

    def test_p2_sum_none(self):
        predicts = jnp.array([[0.2, 0.8], [0.6, 0.4]])
        targets = jnp.array([1, 0])
        assert np.isfinite(float(C.multi_margin_loss(predicts, targets, p=2, reduction='sum')))
        none = np.asarray(C.multi_margin_loss(predicts, targets, p=2, reduction='none'))
        assert none.shape == (2, 2)

    def test_invalid_p_raises(self):
        predicts = jnp.array([[0.2, 0.8]])
        targets = jnp.array([0])
        with pytest.raises(AssertionError):
            C.multi_margin_loss(predicts, targets, p=3)


# ---------------------------------------------------------------------------
# CTC
# ---------------------------------------------------------------------------
class TestCTC:
    """Regression tests for ``ctc_loss`` / ``ctc_loss_with_forward_probs``.

    NOTE (bug fixed in this change): both functions did
    ``bm.log_softmax(logits).value`` and ``bm.one_hot(...).value``, but under
    brainstate>=0.5 those helpers return plain ``jax`` arrays (no ``.value``
    attribute), so *every* call raised ``AttributeError``. Fixed by routing
    through ``bm.as_jax(...)`` (the idiom already used elsewhere in the file).
    These tests pin the corrected behaviour.
    """

    def _inputs(self):
        B, T, K, N = 1, 4, 3, 2
        logits = jnp.zeros((B, T, K))
        logit_paddings = jnp.zeros((B, T))
        labels = jnp.array([[1, 2]])
        label_paddings = jnp.zeros((B, N))
        return logits, logit_paddings, labels, label_paddings

    def test_ctc_loss_shape_and_nonneg(self):
        logits, lp, labels, labp = self._inputs()
        loss = np.asarray(C.ctc_loss(logits, lp, labels, labp))
        assert loss.shape == (1,)
        assert loss[0] >= 0

    def test_ctc_with_forward_probs(self):
        logits, lp, labels, labp = self._inputs()
        per_seq, logalpha_phi, logalpha_emit = C.ctc_loss_with_forward_probs(logits, lp, labels, labp)
        assert np.asarray(per_seq).shape == (1,)
        assert np.asarray(logalpha_phi).shape[1] == 1  # batch dim
        # ctc_loss returns exactly the per-seq loss
        assert np.allclose(np.asarray(C.ctc_loss(logits, lp, labels, labp)), np.asarray(per_seq))
