# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

__all__ = [
  'get_figure',
]


def get_figure(row_num, col_num, row_len=3, col_len=6, name=None):
  """Get the constrained_layout figure.

  Parameters
  ----------
  row_num : int
      The row number of the figure.
  col_num : int
      The column number of the figure.
  row_len : int, float
      The length of each row.
  col_len : int, float
      The length of each column.

  Returns
  -------
  fig_and_gs : tuple
      Figure and GridSpec.
  """
  if name is None:
    fig = plt.figure(figsize=(col_num * col_len, row_num * row_len), constrained_layout=True)
  else:
    fig = plt.figure(name, figsize=(col_num * col_len, row_num * row_len), constrained_layout=True)
  gs = GridSpec(row_num, col_num, figure=fig)
  return fig, gs
