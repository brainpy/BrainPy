# -*- coding: utf-8 -*-
"""Coverage tests for :mod:`brainpy.deprecations`.

Exercises the deprecation helper machinery: ``deprecated`` decorator,
``deprecation_getattr`` (with deprecations, redirects, and missing-attr),
and ``deprecation_getattr2`` (old/new name messaging, ``fn is None`` ->
AttributeError, and missing-attr). Each helper's warning emission and
error branches are covered.
"""
import warnings

import pytest

from brainpy import deprecations as dep


# --------------------------------------------------------------------------- #
# deprecated decorator
# --------------------------------------------------------------------------- #
class TestDeprecatedDecorator:
    def test_emits_warning_and_calls(self):
        @dep.deprecated
        def add(a, b):
            return a + b

        with pytest.warns(DeprecationWarning, match='deprecated function add'):
            result = add(2, 3)
        assert result == 5

    def test_preserves_name(self):
        @dep.deprecated
        def my_fun():
            return 1

        assert my_fun.__name__ == 'my_fun'


# --------------------------------------------------------------------------- #
# deprecation_getattr
# --------------------------------------------------------------------------- #
class TestDeprecationGetattr:
    def test_deprecated_with_fn(self):
        def replacement():
            return 'ok'

        deprecations = {'old_name': ('old_name is gone, use replacement', replacement)}
        get_attr = dep.deprecation_getattr('mymod', deprecations)

        with pytest.warns(DeprecationWarning, match='old_name is gone'):
            fn = get_attr('old_name')
        assert fn is replacement
        assert fn() == 'ok'

    def test_deprecated_with_none_fn_raises(self):
        deprecations = {'removed': ('removed entirely', None)}
        get_attr = dep.deprecation_getattr('mymod', deprecations)
        with pytest.raises(AttributeError, match='removed entirely'):
            get_attr('removed')

    def test_redirect(self):
        import math as redirect_target
        get_attr = dep.deprecation_getattr(
            'mymod', deprecations={}, redirects={'pi': True}, redirect_module=redirect_target
        )
        assert get_attr('pi') == redirect_target.pi

    def test_missing_attribute(self):
        get_attr = dep.deprecation_getattr('mymod', deprecations={})
        with pytest.raises(AttributeError, match="has no attribute 'nope'"):
            get_attr('nope')

    def test_redirects_default_none(self):
        # redirects=None should be coerced to {} without error
        get_attr = dep.deprecation_getattr('mymod', deprecations={})
        with pytest.raises(AttributeError):
            get_attr('whatever')


# --------------------------------------------------------------------------- #
# deprecation_getattr2
# --------------------------------------------------------------------------- #
class TestDeprecationGetattr2:
    def test_old_new_with_fn(self):
        def replacement():
            return 42

        deprecations = {'old': ('bp.old', 'bp.new', replacement)}
        get_attr = dep.deprecation_getattr2('mymod', deprecations)
        with pytest.warns(DeprecationWarning, match='bp.old is deprecated. Use bp.new instead.'):
            fn = get_attr('old')
        assert fn() == 42

    def test_new_name_none_message(self):
        def replacement():
            return 7

        deprecations = {'old': ('bp.old', None, replacement)}
        get_attr = dep.deprecation_getattr2('mymod', deprecations)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fn = get_attr('old')
        assert fn is replacement
        # message should NOT contain "Use ... instead." when new_name is None
        assert any('bp.old is deprecated.' in str(w.message) for w in caught)
        assert not any('instead' in str(w.message) for w in caught)

    def test_fn_none_raises(self):
        deprecations = {'old': ('bp.old', 'bp.new', None)}
        get_attr = dep.deprecation_getattr2('mymod', deprecations)
        with pytest.raises(AttributeError, match='bp.old is deprecated. Use bp.new instead.'):
            get_attr('old')

    def test_missing_attribute(self):
        get_attr = dep.deprecation_getattr2('mymod', deprecations={})
        with pytest.raises(AttributeError, match="has no attribute 'missing'"):
            get_attr('missing')


# --------------------------------------------------------------------------- #
# _deprecate raw helper
# --------------------------------------------------------------------------- #
class TestDeprecateHelper:
    def test_deprecate_emits(self):
        with pytest.warns(DeprecationWarning, match='custom message'):
            dep._deprecate('custom message')
