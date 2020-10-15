
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
- Numba >=
- Sympy
- Matplotlib


Define a Hodgkin–Huxley neuron model
====================================

.. code-block:: python

    import npbrain.numpy as np
    import npbrain as nn

    def HH(geometry, method=None, noise=0., E_Na=50., g_Na=120., E_K=-77.,
           g_K=36., E_Leak=-54.387, g_Leak=0.03, C=1.0, Vr=-65., Vth=20.):

        var2index = {'V': 0, 'm': 1, 'h': 2, 'n': 3}
        num, geometry = nn.format_geometry(geometry)
        state = nn.initial_neu_state(4, num)

        @nn.update(method=method)
        def int_m(m, t, V):
            alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
            beta = 4.0 * np.exp(-(V + 65) / 18)
            return alpha * (1 - m) - beta * m

        @nn.update(method=method)
        def int_h(h, t, V):
            alpha = 0.07 * np.exp(-(V + 65) / 20.)
            beta = 1 / (1 + np.exp(-(V + 35) / 10))
            return alpha * (1 - h) - beta * h

        @nn.update(method=method)
        def int_n(n, t, V):
            alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
            beta = 0.125 * np.exp(-(V + 65) / 80)
            return alpha * (1 - n) - beta * n

        @nn.update(method=method, noise=noise / C)
        def int_V(V, t, Icur, Isyn):
            return (Icur + Isyn) / C

        def update_state(neu_state, t):
            V, Isyn = neu_state[0], neu_state[-1]
            m = nn.clip(int_m(neu_state[1], t, V), 0., 1.)
            h = nn.clip(int_h(neu_state[2], t, V), 0., 1.)
            n = nn.clip(int_n(neu_state[3], t, V), 0., 1.)
            INa = g_Na * m * m * m * h * (V - E_Na)
            IK = g_K * n ** 4 * (V - E_K)
            IL = g_Leak * (V - E_Leak)
            Icur = - INa - IK - IL
            V = int_V(V, t, Icur, Isyn)
            neu_state[0] = V
            neu_state[1] = m
            neu_state[2] = h
            neu_state[3] = n
            nn.judge_spike(neu_state, Vth, t)

        return nn.Neurons(**locals())


Define an AMPA synapse model
============================

.. code-block:: python



.. _How NumpyBrain works: https://numpybrain.readthedocs.io/en/latest/guides/how_it_works.html


