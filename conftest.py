# -*- coding: utf-8 -*-
"""Pytest configuration shared by both test roots (``tests/`` and ``brainpy/``).

Force matplotlib onto the non-interactive ``Agg`` backend so that tests which
exercise the analysis/plotting code paths (e.g. phase-plane and bifurcation
analyses that call ``pyplot.show()``) never try to open a GUI window. This keeps
the suite headless and non-blocking locally and in CI regardless of the
``MPLBACKEND`` environment variable.

Also pin JAX's default matmul precision to ``highest``. On accelerators (notably
NVIDIA GPUs) the default precision uses TF32 for ``float32`` matmuls, which
introduces ~1e-4 relative error. Several correctness tests compare an operator's
full-precision output against a dense ``x @ W`` reference (e.g. the just-in-time
connectivity layers and orthonormality checks); with TF32 those comparisons fail
on GPU while passing on CPU. Pinning the precision makes the suite deterministic
and hardware-independent (CPU already runs at full ``float32``).
"""

import jax
import matplotlib

matplotlib.use('Agg', force=True)
jax.config.update('jax_default_matmul_precision', 'highest')
