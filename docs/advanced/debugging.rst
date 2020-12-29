Debugging
=========

``BrainPy`` supports debugging with `pdb <https://docs.python.org/3/library/pdb.html>`_
module or `breakpoint() <https://docs.python.org/3/library/functions.html#breakpoint>`_.
Currently, we remove the support of debugging in IDEs.

Let's take the HH neuron model as an example to illustrate how to debug your
model within BrainPy.

First import the necessary packages:

.. code-block:: python

    import brainpy as bp
    import numpy as np
    import pdb


Then define the HH neuron model:

.. code-block:: python

    E_Na = 50.
    E_K = -77.
    E_leak = -54.387
    C = 1.0
    g_Na = 120.
    g_K = 36.
    g_leak = 0.03
    V_th = 20.
    noise = 1.

    ST = bp.types.NeuState(
        {'V': -65., 'm': 0.05, 'h': 0.60,
         'n': 0.32, 'spike': 0., 'input': 0.}
    )

    @bp.integrate
    def int_m(m, _t, V):
        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        return alpha * (1 - m) - beta * m

    @bp.integrate
    def int_h(h, _t, V):
        alpha = 0.07 * np.exp(-(V + 65) / 20.)
        beta = 1 / (1 + np.exp(-(V + 35) / 10))
        return alpha * (1 - h) - beta * h

    @bp.integrate
    def int_n(n, _t, V):
        alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
        beta = 0.125 * np.exp(-(V + 65) / 80)
        return alpha * (1 - n) - beta * n

    @bp.integrate
    def int_V(V, _t, m, h, n, I_ext):
        I_Na = (g_Na * np.power(m, 3.0) * h) * (V - E_Na)
        I_K = (g_K * np.power(n, 4.0))* (V - E_K)
        I_leak = g_leak * (V - E_leak)
        dVdt = (- I_Na - I_K - I_leak + I_ext)/C
        return dVdt, noise / C

    def update(ST, _t):
        m = np.clip(int_m(ST['m'], _t, ST['V']), 0., 1.)
        h = np.clip(int_h(ST['h'], _t, ST['V']), 0., 1.)
        n = np.clip(int_n(ST['n'], _t, ST['V']), 0., 1.)
        V = int_V(ST['V'], _t, m, h, n, ST['input'])
        spike = np.logical_and(ST['V'] < V_th, V >= V_th)
        ST['spike'] = spike
        ST['V'] = V
        ST['m'] = m
        ST['h'] = h
        ST['n'] = n
        ST['input'] = 0.

    HH = bp.NeuType(ST=ST,
                    name='HH_neuron',
                    steps=update,
                    mode='vector')



