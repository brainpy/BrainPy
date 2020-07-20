# -*- coding: utf-8 -*-

"""
Visualization toolkit.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import animation
from matplotlib.gridspec import GridSpec


__all__ = [
    'get_figure',

    'mpl_style1',

    'plot_value',
    'plot_potential',
    'plot_raster',
    'animation_potential',
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

###############################
# plotting style
###############################


def mpl_style1(fontsize=22, axes_edgecolor='white', figsize='5,4', lw=1):
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
    rcParams.integrate(params)


###############################
# Neuron and Synapse plotting
###############################


def plot_value(mon, ts, key, val_index=None, ax=None):
    """Show the specified value in the given object (Neurons or Synapses.)

    Parameters
    ----------
    mon : Monitor
        The monitor which record the membrane potential.
    ts : list, numpy.ndarray
        The time steps.
    key : str
        The name of the value to plot.
    val_index : None, int, tuple, list
        The index of the value to plot.
    ax : None, Axes
        The figure to plot.
    """
    if val_index is None:
        val_index = [0]
    elif isinstance(val_index, int):
        val_index = [val_index]
    assert isinstance(val_index, (list, tuple))

    if ax is None:
        ax = plt
    for idx in val_index:
        ax.plot(ts, getattr(mon, key)[:, idx], label='{}-{}'.format(key, idx))
    if len(val_index) > 1:
        ax.legend()


def plot_potential(mon, ts, neuron_index=None, ax=None, label=True, show=False):
    """Show the potential of neurons in the neuron group.

    Parameters
    ----------
    mon : Monitor
        The monitor which record the membrane potential.
    ts : list, numpy.ndarray
        The time steps.
    neuron_index : None, int, tuple, list
        The neuron index to show the potential.
    ax : None, Axes
        The figure to plot.
    label : bool
        Add the xlabel and ylabel.
    show : bool
        Show the figure.
    """
    if neuron_index is None:
        neuron_index = [0]
    elif isinstance(neuron_index, int):
        neuron_index = [neuron_index]
    assert isinstance(neuron_index, (list, tuple))

    if ax is None:
        ax = plt
    for idx in neuron_index:
        ax.plot(ts, mon.V[:, idx], label='N-{}'.format(idx))
    ax.legend()
    if label:
        plt.ylabel('Membrane potential')
        plt.xlabel('Time (ms)')
    if show:
        plt.show()


def plot_raster(mon, times=None, ax=None, markersize=2, color='k', label=True, show=False):
    """Show the rater plot of the spikes.

    Parameters
    ----------
    mon : Monitor
        The monitor which record the spike information.
    times : None, numpy.ndarray
        The run times.
    ax : None, Axes
        The figure.
    markersize : int
        The size of the marker.
    color : str
        The color of the marker.
    label : bool
        Add the xlabel and ylabel.
    show : bool
        Show the figure.
    """

    # get index and time
    if hasattr(mon, 'spike'):  # StateMonitor
        elements = np.where(mon.spike > 0.)
        index = elements[1]
        if hasattr(mon, 'spike_time'):
            time = mon.spike_time[elements]
        else:
            assert times is not None, 'Must provide "times" when StateMonitor has no "spike_time" attribute.'
            time = times[elements[0]]
    else:  # SpikeMonitor
        assert hasattr(mon, 'index'), 'Must be a SpikeMonitor.'
        index = np.array(mon.index)
        time = np.array(mon.time)

    # plot rater
    if ax is None:
        fig, gs = get_figure(1, 1)
        ax = fig.add_subplot(gs[0, 0])
    ax.plot(time, index, '.' + color, markersize=markersize)
    if label:
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron index')
    if show:
        plt.show()


def animation_potential(potentials, size, dt, V_min=0, V_max=10, cmap=None,
                        frame_delay=1., frame_step=1, title_size=10, figsize=None,
                        gif_dpi=None, video_fps=None, save_path=None, show=True):
    """Animate the potentials of the neuron group.

    Parameters
    ----------
    potentials : numpy.ndarray
        The membrane potentials of the neuron group.
    size : tuple
        The size of the neuron group.
    dt : float
        The time duration of each step.
    V_min : float, int
        The minimum of the potential.
    V_max : float, int
        The maximum of the potential.
    cmap : None, str
        The colormap.
    frame_delay : int, float
        The delay to show each frame.
    frame_step : int
        The step to show the potential. If `frame_step=3`, then each
        frame shows one of the every three steps.
    title_size : int
        The size of the title.
    figsize : None, tuple
        The size of the figure.
    gif_dpi : int
        Controls the dots per inch for the movie frames. This combined with
        the figure's size in inches controls the size of the movie. If
        ``None``, use defaults in matplotlib.
    video_fps : int
        Frames per second in the movie. Defaults to ``None``, which will use
        the animation's specified interval to set the frames per second.
    save_path : None, str
        The save path of the animation.
    show : bool
        Whether show the animation.

    Returns
    -------
    figure : plt.figure
        The created figure instance.
    """
    num_step, num_neuron = potentials.shape
    height, width = size

    figsize = figsize or (6, 6)
    fig, gs = get_figure(1, 1, figsize[1], figsize[0])
    fig.add_subplot(gs[0, 0])

    def frame(t):
        img = potentials[t]
        fig.clf()
        plt.pcolor(img, cmap=cmap, vmin=V_min, vmax=V_max)
        plt.colorbar()
        plt.axis('off')
        fig.suptitle("Time: {:.2f} ms".format((t + 1) * dt),
                     fontsize=title_size, fontweight='bold')
        return [fig.gca()]

    potentials = potentials.reshape((num_step, height, width))
    anim_result = animation.FuncAnimation(
        fig, frame, frames=list(range(1, num_step, frame_step)),
        init_func=None, interval=frame_delay, repeat_delay=3000)
    if save_path is None:
        if show:
            plt.show()
    else:
        if save_path[-3:] == 'gif':
            anim_result.save(save_path, dpi=gif_dpi, writer='imagemagick')
        elif save_path[-3:] == 'mp4':
            anim_result.save(save_path, writer='ffmpeg', fps=video_fps, bitrate=3000)
        else:
            anim_result.save(save_path + '.mp4', writer='ffmpeg', fps=video_fps, bitrate=3000)
    return fig

