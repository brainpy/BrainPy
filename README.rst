

.. image:: https://github.com/PKU-NIP-Lab/BrainPy/blob/master/docs/images/logo.png
    :target: https://github.com/PKU-NIP-Lab/BrainPy
    :align: center
    :alt: Logo

.. image:: https://anaconda.org/brainpy/brainpy/badges/license.svg
    :target: https://github.com/PKU-NIP-Lab/BrainPy
    :alt: LICENSE

.. image:: https://readthedocs.org/projects/brainpy/badge/?version=latest
    :target: https://brainpy.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://anaconda.org/brainpy/brainpy/badges/version.svg
    :target: https://anaconda.org/brainpy/brainpy
    :alt: Conda version

.. image:: https://badge.fury.io/py/Brain.Py.svg
    :target: https://badge.fury.io/py/Brain.Py
    :alt: Pypi Version




**Note**: *BrainPy is a project under development.*
*More features are coming soon. Contributions are welcome.*



Why to use BrainPy
=====================

``BrainPy`` is a lightweight framework based on the latest Just-In-Time (JIT)
compilers (especially `Numba <https://numba.pydata.org/>`_).
The goal of ``BrainPy`` is to provide a highly flexible and efficient
neural simulation framework for Python users.
It endows the users with the fully data/logic flow control.
Besides highly flexible, BrainPy is very fast (see the following comparison figure).
Currently, BrainPy supports JIT acceleration on CPU. In future, we will
support GPU acceleration.

.. figure:: https://github.com/PKU-NIP-Lab/NumpyBrain/blob/master/docs/images/speed.png
    :alt: Speed of BrainPy
    :figclass: align-center
    :width: 400px


Installation
============

Install from source code::

    > git clone https://github.com/PKU-NIP-Lab/BrainPy
    > python setup.py install
    >
    > # or
    >
    > pip install git+https://github.com/PKU-NIP-Lab/BrainPy

Install ``BrainPy`` using ``conda``::

    > conda install -c brainpy brainpy

Install ``BrainPy`` using ``pip``::

    > pip install brain.py


The following packages need to be installed to use ``BrainPy``:

- Python >= 3.7
- NumPy >= 1.13
- Sympy >= 1.2
- Matplotlib >= 3.0
- autopep8

Packages recommended to install:

- Numba >= 0.40.0


Define a Hodgkin–Huxley neuron model
====================================

.. code-block:: python

    import brainpy.numpy as np
    import brainpy as bp

    def HH(noise=0., E_Na=50., g_Na=120., E_K=-77., g_K=36.,
           E_Leak=-54.387, g_Leak=0.03, C=1.0, Vth=20.):

        ST = bp.types.NeuState(
            {'V': -65., 'm': 0., 'h': 0., 'n': 0., 'sp': 0., 'inp': 0.},
            help='Hodgkin–Huxley neuron state.\n'
                 '"V" denotes membrane potential.\n'
                 '"n" denotes potassium channel activation probability.\n'
                 '"m" denotes sodium channel activation probability.\n'
                 '"h" denotes sodium channel inactivation probability.\n'
                 '"sp" denotes spiking state.\n'
                 '"inp" denotes synaptic input.\n'
        )

        @bp.integrate
        def int_m(m, t, V):
            alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
            beta = 4.0 * np.exp(-(V + 65) / 18)
            return alpha * (1 - m) - beta * m

        @bp.integrate
        def int_h(h, t, V):
            alpha = 0.07 * np.exp(-(V + 65) / 20.)
            beta = 1 / (1 + np.exp(-(V + 35) / 10))
            return alpha * (1 - h) - beta * h

        @bp.integrate
        def int_n(n, t, V):
            alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
            beta = 0.125 * np.exp(-(V + 65) / 80)
            return alpha * (1 - n) - beta * n

        @bp.integrate
        def int_V(V, t, m, h, n, Isyn):
            INa = g_Na * m ** 3 * h * (V - E_Na)
            IK = g_K * n ** 4 * (V - E_K)
            IL = g_Leak * (V - E_Leak)
            dvdt = (- INa - IK - IL + Isyn) / C
            return (dvdt, noise / C)

        def update(ST, _t_):
            m = np.clip(int_m(ST['m'], _t_, ST['V']), 0., 1.)
            h = np.clip(int_h(ST['h'], _t_, ST['V']), 0., 1.)
            n = np.clip(int_n(ST['n'], _t_, ST['V']), 0., 1.)
            V = int_V(ST['V'], _t_, m, h, n, ST['inp'])
            sp = np.logical_and(ST['V'] < Vth, V >= Vth)
            ST['sp'] = sp
            ST['V'] = V
            ST['m'] = m
            ST['h'] = h
            ST['n'] = n
            ST['inp'] = 0.

        return bp.NeuType(name='HH',
                          requires=dict(ST=ST),
                          steps=update)



Define an AMPA synapse model
============================

.. code-block:: python

    def AMPA(g_max=0.10, E=0., tau_decay=2.0):

        requires = dict(
            ST=bp.types.SynState(
                ['s'], help='AMPA synapse state.'),
            pre=bp.types.NeuState(
                ['sp'], help='Pre-synaptic state must have "sp" item.'),
            post=bp.types.NeuState(
                ['V', 'inp'], help='Post-synaptic neuron must have "V" and "inp" items.')
        )

        @bp.integrate(method='euler')
        def ints(s, t):
            return - s / tau_decay

        def update(ST, _t_, pre):
            s = ints(ST['s'], _t_)
            s += pre['sp']
            ST['s'] = s

        @bp.delayed
        def output(ST, post):
            post_val = - g_max * ST['s'] * (post['V'] - E)
            post['inp'] += post_val

        return bp.SynType(name='AMPA',
                          requires=requires,
                          steps=(update, output),
                          vector_based=False)


