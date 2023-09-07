from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.artist import Artist
from matplotlib.figure import Figure

import brainpy.math as bm

__all__ = [
  'animator',
]


def animator(data, fig, ax, num_steps=False, interval=40, cmap="plasma"):
  """Generate an animation by looping through the first dimension of a
  sample of spiking data.
  Time must be the first dimension of ``data``.

  Example::

      import matplotlib.pyplot as plt

      #  Index into a single sample from a minibatch
      spike_data_sample = bm.random.rand(100, 28, 28)
      print(spike_data_sample.shape)
      >>> (100, 28, 28)

      #  Plot
      fig, ax = plt.subplots()
      anim = splt.animator(spike_data_sample, fig, ax)
      HTML(anim.to_html5_video())

      #  Save as a gif
      anim.save("spike_mnist.gif")

  :param data: Data tensor for a single sample across time steps of
      shape [num_steps x input_size]
  :type data: torch.Tensor

  :param fig: Top level container for all plot elements
  :type fig: matplotlib.figure.Figure

  :param ax: Contains additional figure elements and sets the coordinate
      system. E.g.:
          fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
  :type ax: matplotlib.axes._subplots.AxesSubplot

  :param num_steps: Number of time steps to plot. If not specified,
      the number of entries in the first dimension
          of ``data`` will automatically be used, defaults to ``False``
  :type num_steps: int, optional

  :param interval: Delay between frames in milliseconds, defaults to ``40``
  :type interval: int, optional

  :param cmap: color map, defaults to ``plasma``
  :type cmap: string, optional

  :return: animation to be displayed using ``matplotlib.pyplot.show()``
  :rtype: FuncAnimation

  """

  data = bm.as_numpy(data)
  if not num_steps:
    num_steps = data.shape[0]
  camera = Camera(fig)
  plt.axis("off")
  # iterate over time and take a snapshot with celluloid
  for step in range(
      num_steps
  ):  # im appears unused but is required by camera.snap()
    im = ax.imshow(data[step], cmap=cmap)  # noqa: F841
    camera.snap()
  anim = camera.animate(interval=interval)
  return anim


class Camera:
  """Make animations easier."""

  def __init__(self, figure: Figure) -> None:
    """Create camera from matplotlib figure."""
    self._figure = figure
    # need to keep track off artists for each axis
    self._offsets: Dict[str, Dict[int, int]] = {
      k: defaultdict(int)
      for k in [
        "collections",
        "patches",
        "lines",
        "texts",
        "artists",
        "images",
      ]
    }
    self._photos: List[List[Artist]] = []

  def snap(self) -> List[Artist]:
    """Capture current state of the figure."""
    frame_artists: List[Artist] = []
    for i, axis in enumerate(self._figure.axes):
      if axis.legend_ is not None:
        axis.add_artist(axis.legend_)
      for name in self._offsets:
        new_artists = getattr(axis, name)[self._offsets[name][i]:]
        frame_artists += new_artists
        self._offsets[name][i] += len(new_artists)
    self._photos.append(frame_artists)
    return frame_artists

  def animate(self, *args, **kwargs) -> ArtistAnimation:
    """Animate the snapshots taken.
    Uses matplotlib.animation.ArtistAnimation
    Returns
    -------
    ArtistAnimation
    """
    return ArtistAnimation(self._figure, self._photos, *args, **kwargs)
