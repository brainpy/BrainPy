# -*- coding: utf-8 -*-

import numpy as np

__all__ = [
    'cross_correlation',
    'voltage_fluctuation',
]


###############################
# Neuron synchronization
###############################

def cross_correlation(spikes, bin_size):
    """Calculate cross correlation index between neurons.

    The coherence [1]_ between two neurons i and j is measured by their
    cross-correlation of spike trains at zero time lag within a time bin
    of :math:`\\Delta t = \\tau`. More specifically, suppose that a long
    time interval T is divided into small bins of :math:`\\Delta t` and
    that two spike trains are given by :math:`X(l)=` 0 or 1, :math:`Y(l)=` 0
    or 1, :math:`l=1,2, \\ldots, K(T / K=\\tau)`. Thus, we define a coherence
    measure for the pair as:

    .. math::

        \\kappa_{i j}(\\tau)=\\frac{\\sum_{l=1}^{K} X(l) Y(l)}
        {\\sqrt{\\sum_{l=1}^{K} X(l) \\sum_{l=1}^{K} Y(l)}}

    The population coherence measure :math:`\\kappa(\\tau)` is defined by the
    average of :math:`\\kappa_{i j}(\\tau)` over many pairs of neurons in the
    network.

    Parameters
    ----------
    spikes : numpy.ndarray
        The history of spike states of the neuron group.
        It can be easily get via `StateMonitor(neu, ['spike'])`.
    bin_size : int
        The bin size to normalize spike states.

    Returns
    -------
    cc_index : float
        The cross correlation value which represents the synchronization index.

    References
    ----------
    .. [1] Wang, Xiao-Jing, and György Buzsáki. "Gamma oscillation by synaptic
           inhibition in a hippocampal interneuronal network model." Journal of
           neuroscience 16.20 (1996): 6402-6413.
    """

    num_hist, num_neu = spikes.shape
    num_bin = int(np.ceil(num_hist / bin_size))
    if num_bin * bin_size != num_hist:
        spikes = np.append(spikes, np.zeros(num_bin * num_hist - num_hist, num_neu), axis=0)
    states = spikes.T.reshape(num_neu, num_bin, bin_size)
    states = np.float64(np.sum(states, axis=2) > 0.)

    all_k = []
    for i in range(num_neu):
        for j in range(i + 1, num_neu):
            sqrt_ij = np.sqrt(np.sum(states[i]) * np.sum(states[j]))
            k = 0. if sqrt_ij == 0. else np.sum(states[i] * states[j]) / sqrt_ij
            all_k.append(k)
    return np.mean(all_k)


def voltage_fluctuation(potentials):
    """Calculate neuronal synchronization via voltage variance.

    The method comes from [1]_ [2]_ [3]_.

    First, average over the membrane potential :math:`V`

    .. math::

        V(t) = \\frac{1}{N} \\sum_{i=1}^{N} V_i(t)

    The variance of the time fluctuations of :math:`V(t)` is

    .. math::

        \\sigma_V^2 = \\left\\langle \\left[ V(t) \\right]^2 \\right\\rangle_t -
        \\left[ \\left\\langle V(t) \\right\\rangle_t \\right]^2

    where :math:`\\left\\langle \\ldots \\right\\rangle_t = (1 / T_m) \\int_0^{T_m} dt \\, \\ldots`
    denotes time-averaging over a large time, :math:`\\tau_m`. After normalization
    of :math:`\\sigma_V` to the average over the population of the single cell
    membrane potentials

    .. math::

        \\sigma_{V_i}^2 = \\left\\langle\\left[ V_i(t) \\right]^2 \\right\\rangle_t -
        \\left[ \\left\\langle V_i(t) \\right\\rangle_t \\right]^2

    one defines a synchrony measure, :math:`\\chi (N)`, for the activity of a system
    of :math:`N` neurons by:

    .. math::

        \\chi^2 \\left( N \\right) = \\frac{\\sigma_V^2}{ \\frac{1}{N} \\sum_{i=1}^N
        \\sigma_{V_i}^2}

    Parameters
    ----------
    potentials : numpy.ndarray
        The membrane potentials of the neuron group, which can be easily accessed by
        `StateMonitor(neu, ['V'])`.

    Returns
    -------
    sync_index : float
        The synchronization index.

    References
    ----------
    .. [1] Golomb, D. and Rinzel J. (1993) Dynamics of globally coupled
           inhibitory neurons with heterogeneity. Phys. Rev. E 48:4810-4814.
    .. [2] Golomb D. and Rinzel J. (1994) Clustering in globally coupled
           inhibitory neurons. Physica D 72:259-282.
    .. [3] David Golomb (2007) Neuronal synchrony measures. Scholarpedia, 2(1):1347.
    """

    num_hist, num_neu = potentials.shape
    avg = np.mean(potentials, axis=1)
    avg_var = np.mean(avg * avg) - np.mean(avg) ** 2
    neu_vars = []
    for i in range(num_neu):
        neu = potentials[:, i]
        neu_vars.append(np.mean(neu * neu) - np.mean(neu) ** 2)
    var_mean = np.mean(neu_vars)
    return avg_var / var_mean if var_mean != 0. else 1.
