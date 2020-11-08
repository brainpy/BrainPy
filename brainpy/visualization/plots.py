# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec

from .. import numpy as np
from ..errors import ModelUseError

__all__ = [
    'line_plot',
    'raster_plot',
    'animate_potential',
]


def line_plot(ts,
              val_matrix,
              plot_ids=None,
              ax=None,
              xlim=None,
              ylim=None,
              xlabel='Time (ms)',
              ylabel='value',
              legend_prefix='N',
              show=False):
    """Show the specified value in the given object (Neurons or Synapses.)

    Parameters
    ----------
    ts : a_list, numpy.ndarray
        The time steps.
    val_matrix : np.ndarray
        The value matrix which record the history trajectory.
        It can be easily accessed by specifying the ``monitors``
        of NeuGroup/SynConn by:
        ``neu/syn = NeuGroup/SynConn(..., monitors=[k1, k2])``
    plot_ids : None, int, tuple, a_list
        The index of the value to plot.
    ax : None, Axes
        The figure to plot.
    xlim : list, tuple
        The xlim.
    ylim : list, tuple
        The ylim.
    xlabel : str
        The xlabel.
    ylabel : str
        The ylabel.
    legend_prefix : str
        The prefix of legend for plot.
    show : bool
        Whether show the figure.
    """
    # get plot_ids
    if plot_ids is None:
        plot_ids = [0]
    elif isinstance(plot_ids, int):
        plot_ids = [plot_ids]
    try:
        assert isinstance(plot_ids, (list, tuple))
    except AssertionError:
        raise ModelUseError('"plot_ids" specifies the value index to plot, '
                            'it must be a list/tuple.')

    # get ax
    if ax is None:
        plt.figure()
        ax = plt

    # plot
    for idx in plot_ids:
        ax.plot(ts, val_matrix[:, idx], label=f'{legend_prefix}-{idx}')

    # legend
    if len(plot_ids) > 1:
        ax.legend()

    # xlim
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])

    # ylim
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    # xlable
    if xlabel:
        plt.xlabel(xlabel)

    # ylabel
    if ylabel:
        plt.ylabel(ylabel)

    # show
    if show:
        plt.show()


def raster_plot(ts,
                sp_matrix,
                ax=None,
                markersize=2,
                color='k',
                xlabel='Time (ms)',
                ylabel='Neuron index',
                xlim=None,
                ylim=None,
                show=False):
    """Show the rater plot of the spikes.

    Parameters
    ----------
    ts : numpy.ndarray
        The run times.
    sp_matrix : np.ndarray
        The spike matrix which records the spike information.
        It can be easily accessed by specifying the ``monitors``
        of NeuGroup by: ``neu = NeuGroup(..., monitors=['spike'])``
    ax : Axes
        The figure.
    markersize : int
        The size of the marker.
    color : str
        The color of the marker.
    xlim : list, tuple
        The xlim.
    ylim : list, tuple
        The ylim.
    xlabel : str
        The xlabel.
    ylabel : str
        The ylabel.
    show : bool
        Show the figure.
    """
    # get index and time
    elements = np.where(sp_matrix > 0.)
    index = elements[1]
    time = ts[elements[0]]

    # plot rater
    if ax is None:
        ax = plt
    ax.plot(time, index, '.' + color, markersize=markersize)

    # xlable
    if xlabel:
        plt.xlabel(xlabel)

    # ylabel
    if ylabel:
        plt.ylabel(ylabel)

    if xlim:
        plt.xlim(xlim[0], xlim[1])

    if ylim:
        plt.ylim(ylim[0], ylim[1])

    if show:
        plt.show()


def animate_potential(potentials, size, dt, min=None, max=None, cmap=None,
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
    min : float, int
        The minimum of the potential.
    max : float, int
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
    min = potentials.min() if min is None else min
    max = potentials.max() if max is None else max

    figsize = figsize or (6, 6)

    fig = plt.figure(figsize=(figsize[0], figsize[1]), constrained_layout=True)
    gs = GridSpec(1, 1, figure=fig)
    fig.add_subplot(gs[0, 0])

    def frame(t):
        img = potentials[t]
        fig.clf()
        plt.pcolor(img, cmap=cmap, vmin=min, vmax=max)
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
