# -*- coding: utf-8 -*-


from matplotlib import rcParams

__all__ = [
  'plot_style1',
]


def plot_style1(fontsize=22, axes_edgecolor='black', figsize='5,4', lw=1):
  """Plot style for publication.

  Parameters
  ----------
  fontsize : int
      The font size.
  axes_edgecolor : str
      The exes edge color.
  figsize : str, tuple
      The figure size.
  lw : int
      Line width.
  """
  rcParams['text.latex.preamble'] = [r"\usepackage{amsmath, lmodern}"]
  params = {
    'text.usetex': True,
    'font.family': 'lmodern',
    # 'text.latex.unicode': True,
    'text.color': 'black',
    'xtick.labelsize': fontsize - 2,
    'ytick.labelsize': fontsize - 2,
    'axes.labelsize': fontsize,
    'axes.labelweight': 'bold',
    'axes.edgecolor': axes_edgecolor,
    'axes.titlesize': fontsize,
    'axes.titleweight': 'bold',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.grid': False,
    'axes.facecolor': 'white',
    'lines.linewidth': lw,
    "figure.figsize": figsize,
  }
  rcParams.update(params)
