# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

__all__ = [
    'get_figure',
]


def get_figure(n_row, n_col, len_row=3, len_col=6):
    """Get the constrained_layout figure.

    Parameters
    ----------
    n_row : int
        The row number of the figure.
    n_col : int
        The column number of the figure.
    len_row : int, float
        The length of each row.
    len_col : int, float
        The length of each column.

    Returns
    -------
    fig_and_gs : tuple
        Figure and GridSpec.
    """
    fig = plt.figure(figsize=(n_col * len_col, n_row * len_row), constrained_layout=True)
    gs = GridSpec(n_row, n_col, figure=fig)
    return fig, gs
