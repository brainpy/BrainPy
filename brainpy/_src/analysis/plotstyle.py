# -*- coding: utf-8 -*-


__all__ = [
  'plot_schema',
  'set_plot_schema',
  'set_markersize',
]

from .stability import (CENTER_MANIFOLD, SADDLE_NODE, STABLE_POINT_1D,
                        UNSTABLE_POINT_1D, CENTER_2D, STABLE_NODE_2D,
                        STABLE_FOCUS_2D, STABLE_STAR_2D, STABLE_DEGENERATE_2D,
                        UNSTABLE_NODE_2D, UNSTABLE_FOCUS_2D, UNSTABLE_STAR_2D,
                        UNSTABLE_DEGENERATE_2D, UNSTABLE_LINE_2D,
                        STABLE_POINT_3D, UNSTABLE_POINT_3D, STABLE_NODE_3D, 
                        UNSTABLE_SADDLE_3D, UNSTABLE_NODE_3D, STABLE_FOCUS_3D, 
                        UNSTABLE_FOCUS_3D, UNSTABLE_CENTER_3D, UNKNOWN_3D)


_markersize = 10

plot_schema = {}

plot_schema[CENTER_MANIFOLD] = {'color': 'orangered', 'markersize': _markersize, 'linestyle': 'None', 'marker': '.'}
plot_schema[SADDLE_NODE] = {"color": 'tab:blue', 'markersize': _markersize, 'linestyle': 'None', 'marker': '.'}

plot_schema[STABLE_POINT_1D] = {"color": 'tab:red', 'markersize': _markersize, 'linestyle': 'None', 'marker': '.'}
plot_schema[UNSTABLE_POINT_1D] = {"color": 'tab:olive', 'markersize': _markersize, 'linestyle': 'None', 'marker': '.'}

plot_schema.update({
  CENTER_2D: {'color': 'lime', 'markersize': _markersize, 'linestyle': 'None', 'marker': '.'},
  STABLE_NODE_2D: {"color": 'tab:red', 'markersize': _markersize, 'linestyle': 'None', 'marker': '.'},
  STABLE_FOCUS_2D: {"color": 'tab:purple', 'markersize': _markersize, 'linestyle': 'None', 'marker': '.'},
  STABLE_STAR_2D: {'color': 'tab:olive', 'markersize': _markersize, 'linestyle': 'None', 'marker': '.'},
  STABLE_DEGENERATE_2D: {'color': 'blueviolet', 'markersize': _markersize, 'linestyle': 'None', 'marker': '.'},
  UNSTABLE_NODE_2D: {"color": 'tab:orange', 'markersize': _markersize, 'linestyle': 'None', 'marker': '.'},
  UNSTABLE_FOCUS_2D: {"color": 'tab:cyan', 'markersize': _markersize, 'linestyle': 'None', 'marker': '.'},
  UNSTABLE_STAR_2D: {'color': 'green', 'markersize': _markersize, 'linestyle': 'None', 'marker': '.'},
  UNSTABLE_DEGENERATE_2D: {'color': 'springgreen', 'markersize': _markersize, 'linestyle': 'None', 'marker': '.'},
  UNSTABLE_LINE_2D: {'color': 'dodgerblue', 'markersize': _markersize, 'linestyle': 'None', 'marker': '.'},
})


plot_schema.update({
  STABLE_POINT_3D: {'color': 'tab:gray', 'markersize': _markersize, 'linestyle': 'None', 'marker': '.'},
  UNSTABLE_POINT_3D: {'color': 'tab:purple', 'markersize': _markersize, 'linestyle': 'None', 'marker': '.'},
  STABLE_NODE_3D: {'color': 'tab:green', 'markersize': _markersize, 'linestyle': 'None', 'marker': '.'},
  UNSTABLE_SADDLE_3D: {'color': 'tab:red', 'markersize': _markersize, 'linestyle': 'None', 'marker': '.'},
  UNSTABLE_FOCUS_3D: {'color': 'tab:pink', 'markersize': _markersize, 'linestyle': 'None', 'marker': '.'},
  STABLE_FOCUS_3D: {'color': 'tab:purple', 'markersize': _markersize, 'linestyle': 'None', 'marker': '.'},
  UNSTABLE_NODE_3D: {'color': 'tab:orange', 'markersize': _markersize, 'linestyle': 'None', 'marker': '.'},
  UNSTABLE_CENTER_3D: {'color': 'tab:olive', 'markersize': _markersize, 'linestyle': 'None', 'marker': '.'},
  UNKNOWN_3D: {'color': 'tab:cyan', 'markersize': _markersize, 'linestyle': 'None', 'marker': '.'},
})


def set_plot_schema(fixed_point: str, **schema):
  if not isinstance(fixed_point, str):
    raise TypeError(f'Must instance of string, but we got {type(fixed_point)}: {fixed_point}')
  if fixed_point not in plot_schema:
    raise KeyError(f'Fixed point type {fixed_point} does not found in the built-in types. ')
  plot_schema[fixed_point].update(**schema)


def set_markersize(markersize):
  if not isinstance(markersize, int):
    raise TypeError(f"Must be an integer, but got {type(markersize)}: {markersize}")
  global _markersize
  __markersize = markersize
  for key in tuple(plot_schema.keys()):
    plot_schema[key]['markersize'] = markersize


