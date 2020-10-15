
.. image:: https://github.com/PKU-NIP-Lab/NumpyBrain/blob/master/docs/images/logo.png
    :target: https://github.com/PKU-NIP-Lab/NumpyBrain
    :align: center
    :alt: logo

.. image:: https://readthedocs.org/projects/numpybrain/badge/?version=latest
    :target: https://numpybrain.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://anaconda.org/oujago/npbrain/badges/version.svg
    :target: https://anaconda.org/oujago/npbrain

.. image:: https://badge.fury.io/py/npbrain.svg
    :target: https://badge.fury.io/py/npbrain



**Note**: *NumpyBrain is a project under development.*
*More features are coming soon. Contributions are welcome.*


Why to use NumpyBrain
=====================

``NumpyBrain`` is a microkernel framework for SNN (spiking neural network) simulation
purely based on **native** python. It only relies on `NumPy <https://numpy.org/>`_.
However, if you want to get faster performance,you can additionally
install `Numba <http://numba.pydata.org/>`_. With `Numba`, the speed of C or FORTRAN can
be obtained in the simulation.

``NumpyBrain`` wants to provide a highly flexible and efficient SNN simulation
framework for Python users. It endows the users with the fully data/logic flow control.
The core of the framework is a micro-kernel, and it's easy to understand (see
`How NumpyBrain works`_).
Based on the kernel, the extension of the new models or the customization of the
data/logic flows are very simple for users. Ample examples (such as LIF neuron,
HH neuron, or AMPA synapse, GABA synapse and GapJunction) are also provided.
Besides the consideration of **flexibility**, for accelerating the running
**speed** of NumPy codes, `Numba` is used. For most of the times,
models running on `Numba` backend is very fast
(see `examples/benchmark <https://github.com/PKU-NIP-Lab/NumpyBrain/tree/master/examples/benchmark>`_).

.. figure:: https://github.com/PKU-NIP-Lab/NumpyBrain/blob/master/docs/images/speed_comparison.png
    :alt: Speed comparison with brian2
    :figclass: align-center
    :width: 350px

More details about NumpyBrain please see our `document <https://numpybrain.readthedocs.io/en/latest/>`_.


Installation
============

Install ``NumpyBrain`` using ``pip``::

    $> pip install git+https://github.com/PKU-NIP-Lab/NumpyBrain

Install from source code::

    $> python setup.py install


The following packages need to be installed to use ``NumpyBrain``:

- Python >= 3.5
- NumPy >= 1.13
- Numba >= 0.40.0
- Sympy >= 1.2
- Matplotlib >= 2.0
- autopep8


Define a Hodgkin–Huxley neuron model
====================================

.. code-block:: python

    import npbrain.numpy as np
    import npbrain as nb

    def HH(noise=0., E_Na=50., g_Na=120., E_K=-77., g_K=36.,
           E_Leak=-54.387, g_Leak=0.03, C=1.0, Vth=20.):

        ST = nb.types.NeuState(
            {'V': -65., 'm': 0., 'h': 0., 'n': 0., 'sp': 0., 'inp': 0.},
            help='Hodgkin–Huxley neuron state.\n'
                 '"V" denotes membrane potential.\n'
                 '"n" denotes potassium channel activation probability.\n'
                 '"m" denotes sodium channel activation probability.\n'
                 '"h" denotes sodium channel inactivation probability.\n'
                 '"sp" denotes spiking state.\n'
                 '"inp" denotes synaptic input.\n'
        )

        @nb.integrate
        def int_m(m, t, V):
            alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
            beta = 4.0 * np.exp(-(V + 65) / 18)
            return alpha * (1 - m) - beta * m

        @nb.integrate
        def int_h(h, t, V):
            alpha = 0.07 * np.exp(-(V + 65) / 20.)
            beta = 1 / (1 + np.exp(-(V + 35) / 10))
            return alpha * (1 - h) - beta * h

        @nb.integrate
        def int_n(n, t, V):
            alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
            beta = 0.125 * np.exp(-(V + 65) / 80)
            return alpha * (1 - n) - beta * n

        @nb.integrate(noise=noise / C)
        def int_V(V, t, m, h, n, Isyn):
            INa = g_Na * m ** 3 * h * (V - E_Na)
            IK = g_K * n ** 4 * (V - E_K)
            IL = g_Leak * (V - E_Leak)
            dvdt = (- INa - IK - IL + Isyn) / C
            return dvdt

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

        return nb.NeuType(requires={"ST": ST}, steps=update, vector_based=True)



Define an AMPA synapse model
============================

.. code-block:: python

    def AMPA(g_max=0.10, E=0., tau_decay=2.0):

        requires = dict(
            ST=nb.types.SynState(['s'], help='AMPA synapse state.'),
            pre=nb.types.NeuState(['sp'], help='Pre-synaptic neuron state must have "sp" item.'),
            post=nb.types.NeuState(['V', 'inp'], help='Pre-synaptic neuron state must have "V" and "inp" item.'),
        )

        @nb.integrate(method='euler')
        def ints(s, t):
            return - s / tau_decay

        @nb.delay_push
        def update(ST, _t_, pre):
            s = ints(ST['s'], _t_)
            s += pre['sp']
            ST['s'] = s

        @nb.delay_pull
        def output(ST, post):
            post_val = - g_max * ST['s'] * (post['V'] - E)
            post['inp'] += post_val

        return nb.SynType(requires=requires, steps=(update, output), vector_based=False)



.. _How NumpyBrain works: https://numpybrain.readthedocs.io/en/latest/guides/how_it_works.html


