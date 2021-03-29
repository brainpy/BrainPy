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


backend_missing_msg = '"{bk}" must be installed when users want to set {bk} backend. \n' \
                      'Please install {bk} through "pip install {bk}" or "conda install {bk}".'
