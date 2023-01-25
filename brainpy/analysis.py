# -*- coding: utf-8 -*-


from brainpy._src.analysis.lowdim.lowdim_phase_plane import (
  PhasePlane1D as PhasePlane1D,
  PhasePlane2D as PhasePlane2D,
)

from brainpy._src.analysis.lowdim.lowdim_bifurcation import (
  Bifurcation1D as Bifurcation1D,
  Bifurcation2D as Bifurcation2D,
  FastSlow1D as FastSlow1D,
  FastSlow2D as FastSlow2D,
)

from brainpy._src.analysis.highdim.slow_points import (
  SlowPointFinder as SlowPointFinder,
)

from brainpy._src.analysis.constants import (CONTINUOUS as CONTINUOUS,
                                             DISCRETE as DISCRETE)

from brainpy._src.analysis import plotstyle, stability, constants
C = constants

