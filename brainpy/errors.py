# -*- coding: utf-8 -*-


class ModelDefError(Exception):
    """Model definition error."""
    pass


class ModelUseError(Exception):
    """Model use error."""
    pass


class DiffEqError(Exception):
    pass


class CodeError(Exception):
    pass


class AnalyzerError(Exception):
    pass


class PackageMissingError(Exception):
    pass

