# -*- coding: utf-8 -*-


__all__ = [
  'visualize'
]


class visualize(object):
  @staticmethod
  def get_figure(row_num,
                 col_num,
                 row_len=3,
                 col_len=6):
    from .figures import get_figure
    return get_figure(row_num, col_num, row_len, col_len)

  @staticmethod
  def line_plot(ts,
                val_matrix,
                plot_ids=None,
                ax=None,
                xlim=None,
                ylim=None,
                xlabel='Time (ms)',
                ylabel=None,
                legend=None,
                title=None,
                show=False,
                **kwargs):
    from .plots import line_plot
    line_plot(ts, val_matrix, plot_ids=plot_ids, ax=ax, xlim=xlim, ylim=ylim,
              xlabel=xlabel, ylabel=ylabel, legend=legend, title=title, show=show, **kwargs)

  @staticmethod
  def raster_plot(ts,
                  sp_matrix,
                  ax=None,
                  marker='.',
                  markersize=2,
                  color='k',
                  xlabel='Time (ms)',
                  ylabel='Neuron index',
                  xlim=None,
                  ylim=None,
                  title=None,
                  show=False,
                  **kwargs):
    from .plots import raster_plot
    raster_plot(ts, sp_matrix, ax=ax, marker=marker, markersize=markersize, color=color,
                xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, title=title, show=show, **kwargs)

  @staticmethod
  def animate_1D(dynamical_vars,
                 static_vars=(),
                 dt=None,
                 xlim=None,
                 ylim=None,
                 xlabel=None,
                 ylabel=None,
                 frame_delay=50.,
                 frame_step=1,
                 title_size=10,
                 figsize=None,
                 gif_dpi=None,
                 video_fps=None,
                 save_path=None,
                 show=True,
                 **kwargs):
    from .plots import animate_1D
    return animate_1D(dynamical_vars, static_vars=static_vars, dt=dt, xlim=xlim, ylim=ylim,
                      xlabel=xlabel, ylabel=ylabel, frame_delay=frame_delay, frame_step=frame_step,
                      title_size=title_size, figsize=figsize, gif_dpi=gif_dpi, video_fps=video_fps,
                      save_path=save_path, show=show, **kwargs)

  @staticmethod
  def animate_2D(values,
                 net_size,
                 dt=None,
                 val_min=None,
                 val_max=None,
                 cmap=None,
                 frame_delay=10,
                 frame_step=1,
                 title_size=10,
                 figsize=None,
                 gif_dpi=None,
                 video_fps=None,
                 save_path=None,
                 show=True):
    from .plots import animate_2D
    return animate_2D(values, net_size, dt=dt, val_min=val_min, val_max=val_max, cmap=cmap,
                      frame_delay=frame_delay, frame_step=frame_step, title_size=title_size,
                      figsize=figsize, gif_dpi=gif_dpi, video_fps=video_fps, save_path=save_path, show=show)

  @staticmethod
  def remove_axis(ax, *pos):
    from .plots import remove_axis
    return remove_axis(ax, *pos)

  @staticmethod
  def plot_style1(fontsize=22,
                  axes_edgecolor='black',
                  figsize='5,4',
                  lw=1):
    from .styles import plot_style1
    plot_style1(fontsize=fontsize, axes_edgecolor=axes_edgecolor, figsize=figsize, lw=lw)
