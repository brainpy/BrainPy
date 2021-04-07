# -*- coding: utf-8 -*-


class ModelDefError(Exception):
    """Model definition error."""
    pass


class ModelUseError(Exception):
    """Model use error."""
    pass


class DiffEqError(Exception):
    """The differential equation definition error.
    """
    pass


class CodeError(Exception):
    """Code definition error.
    """
    pass


class AnalyzerError(Exception):
    """Differential equation analyzer error.
    """


class PackageMissingError(Exception):
    """The package missing error.
    """
    pass


class BackendNotInstalled(Exception):
    def __init__(self, backend):
        super(BackendNotInstalled, self).__init__('"{bk}" must be installed when users want '
                                                  'to set {bk} backend. \n' 
                                                  'Please install {bk} through "pip install {bk}" '
                                                  'or "conda install {bk}".'.format(bk=backend))

