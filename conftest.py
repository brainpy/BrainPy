# -*- coding: utf-8 -*-
"""Pytest configuration shared by both test roots (``tests/`` and ``brainpy/``).

Force matplotlib onto the non-interactive ``Agg`` backend so that tests which
exercise the analysis/plotting code paths (e.g. phase-plane and bifurcation
analyses that call ``pyplot.show()``) never try to open a GUI window. This keeps
the suite headless and non-blocking locally and in CI regardless of the
``MPLBACKEND`` environment variable.
"""

import matplotlib

matplotlib.use('Agg', force=True)
