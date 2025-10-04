# -*- coding: utf-8 -*-


import braintools

__all__ = [
    'cross_correlation',
    'voltage_fluctuation',
    'matrix_correlation',
    'weighted_correlation',
    'functional_connectivity',
    # 'functional_connectivity_dynamics',
]

cross_correlation = braintools.metric.cross_correlation
voltage_fluctuation = braintools.metric.voltage_fluctuation
matrix_correlation = braintools.metric.matrix_correlation
functional_connectivity = braintools.metric.functional_connectivity
weighted_correlation = braintools.metric.weighted_correlation
