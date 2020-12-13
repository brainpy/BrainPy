# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Usage of `inputs` module

# +
import numpy as np
import matplotlib.pyplot as plt

import brainpy as bp
# -

# ## constant_current

# `constant_current()` function helps you to format constant current in several periods.
#
# For example, if you want to get an input in which 0-100 ms is zero, 100-400 ms is value `1.`, 
# and 400-500 ms is zero, then, you can define:

# +
current, duration = bp.inputs.constant_current([(0, 100), (1, 300), (0, 100)], 0.1)

fig, gs = bp.visualize.get_figure(1, 1)
fig.add_subplot(gs[0, 0])
ts = np.arange(0, duration, 0.1)
plt.plot(ts, current)
plt.title('[(0, 100), (1, 300), (0, 100)]')
plt.show()
# -
# Another example is this:

# +
current, duration = bp.inputs.constant_current([(-1, 10), (1, 3), (3, 30), (-0.5, 10)], 0.1)

fig, gs = bp.visualize.get_figure(1, 1)
fig.add_subplot(gs[0, 0])
ts = np.arange(0, duration, 0.1)
plt.plot(ts, current)
plt.title('[(-1, 10), (1, 3), (3, 30), (-0.5, 10)]')
plt.show()
# -

# ## spike_current

# `spike_current()` function helps you to construct an input like a series of short-time spikes.

# +
points, length, size, duration, _dt = [10, 20, 30, 200, 300], 1., 0.5, 1000, 0.1
current = bp.inputs.spike_current(points, length, size, duration, _dt)

fig, gs = bp.visualize.get_figure(1, 1)
fig.add_subplot(gs[0, 0])
ts = np.arange(0, duration, _dt)
plt.plot(ts, current)
plt.title(r'points=%s, duration=%d' % (points, duration))
plt.show()
# -

# In the above example, at 10 ms, 20 ms, 30 ms, 200 ms, 300 ms, the assumed neuron produces spikes. Each spike 
# lasts 1 ms, and the spike current is 0.5.

# ## ramp_current

# +
fig, gs = bp.visualize.get_figure(2, 1)

duration, _dt = 1000, 0.1
current = bp.inputs.ramp_current(0, 1, duration)

ts = np.arange(0, duration, _dt)
fig.add_subplot(gs[0, 0])
plt.plot(ts, current)
plt.title(r'$c_{start}$=0, $c_{end}$=%d, duration, dt=%.1f, '
          r'$t_{start}$=0, $t_{end}$=None' % (duration, _dt,))

duration, _dt, t_start, t_end = 1000, 0.1, 200, 800
current = bp.inputs.ramp_current(0, 1, duration, t_start, t_end)

ts = np.arange(0, duration, _dt)
fig.add_subplot(gs[1, 0])
plt.plot(ts, current)
plt.title(r'$c_{start}$=0, $c_{end}$=1, duration=%d, dt=%.1f, '
          r'$t_{start}$=%d, $t_{end}$=%d' % (duration, _dt, t_start, t_end))

plt.show()

