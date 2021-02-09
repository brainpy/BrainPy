# -*- coding: utf-8 -*-


class ModelDefError(Exception):
    """Model definition error."""
    pass


class ModelUseError(Exception):
    """Model use error."""
    pass


class TypeMismatchError(Exception):
    pass


class IntegratorError(Exception):
    pass


class DiffEquationError(Exception):
    pass


class CodeError(Exception):
    pass


class AnalyzerError(Exception):
    pass
