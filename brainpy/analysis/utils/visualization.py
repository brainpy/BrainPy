# -*- coding: utf-8 -*-

import numpy as np


def add_arrow(line, position=None, direction='right', size=15, color=None):
  """
  add an arrow to a line.

  line:       Line2D object
  position:   x-position of the arrow. If None, mean of xdata is taken
  direction:  'left' or 'right'
  size:       size of the arrow in fontsize points
  color:      if None, line color is taken.
  """
  if color is None:
    color = line.get_color()

  xdata = line.get_xdata()
  ydata = line.get_ydata()

  if position is None:
    position = xdata.mean()
  # find closest index
  start_ind = np.argmin(np.absolute(xdata - position))
  if direction == 'right':
    end_ind = start_ind + 1
  else:
    end_ind = start_ind - 1

  line.axes.annotate(text='',
                     xytext=(xdata[start_ind], ydata[start_ind]),
                     xy=(xdata[end_ind], ydata[end_ind]),
                     arrowprops=dict(arrowstyle="->", color=color),
                     size=size)
