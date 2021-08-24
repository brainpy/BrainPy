# -*- coding: utf-8 -*-


class RunningError(Exception):
  """The error occurred in the running function."""
  pass


class ModelDefError(Exception):
  """Model definition error."""
  pass


class ModelUseError(Exception):
  """Model use error."""
  pass


class IntegratorError(Exception):
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
    super(BackendNotInstalled, self).__init__(
      '"{bk}" must be installed when the user wants to use {bk} backend. \n'
      'Please install {bk} through "pip install {bk}" '
      'or "conda install {bk}".'.format(bk=backend))


class UniqueNameError(Exception):
  def __init__(self, *args):
    super(UniqueNameError, self).__init__(*args)


class UnsupportedError(Exception):
  pass