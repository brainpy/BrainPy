
.. image:: https://github.com/chaoming0625/NumpyBrain/blob/master/docs/images/logo.png
    :target: https://github.com/chaoming0625/NumpyBrain
    :align: center
    :alt: logo

.. image:: https://readthedocs.org/projects/numpybrain/badge/?version=latest
    :target: https://numpybrain.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
    :target: https://github.com/chaoming0625/NumpyBrain/blob/master/LICENSE

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
(see `examples/benchmark <https://github.com/chaoming0625/NumpyBrain/tree/master/examples/benchmark>`_).

.. figure:: https://github.com/chaoming0625/NumpyBrain/blob/master/docs/images/speed_comparison.png
    :alt: Speed comparison with brian2
    :figclass: align-center
    :width: 350px

More details about NumpyBrain please see our `document <https://numpybrain.readthedocs.io/en/latest/>`_.


Installation
============

Install ``NumpyBrain`` using ``pip``::

    $> pip install npbrain
    $> # or
    $> pip install git+https://github.com/chaoming0625/NumpyBrain

Install ``NumpyBrain`` using ``conda``::

    $> conda install -c oujago npbrain

Install from source code::

    $> python setup.py install


The following packages need to be installed to use ``NumpyBrain``:

- Python >= 3.5
- NumPy
- Numba
- Matplotlib


Getting started: 30 seconds to NumpyBrain
=========================================

First of all, import the package, and set the numerical backend you prefer:

.. code-block:: python

    import numpy as np
    import npbrain as nn

    nn.profile.set_backend('numba')  # or "numpy"

Next, define two neuron groups:

.. code-block:: python

    lif1 = nn.LIF(500, noise=0.5, method='Ito_milstein')  # or method='euler'
    lif2 = nn.LIF(1000, noise=1.1, method='Ito_milstein')

Then, create one ``Synapse`` to connect them both.

.. code-block:: python

    conn = nn.connect.fixed_prob(lif1.num, lif2.num, prob=0.2)
    syn = nn.VoltageJumpSynapse(lif1, lif2, weights=0.2, connection=conn)

In order to inspect the dynamics of two ``LIF`` neuron groups, we use ``StateMonitor``
to record the membrane potential and the spiking events.

.. code-block:: python

    mon_lif1 = nn.StateMonitor(lif1, ['V', 'spike'])
    mon_lif2 = nn.StateMonitor(lif2, ['V', 'spike'])

All above definitions help us to construct a **network**. Providing the name of the
simulation object (for example, ``mon1=mon_lif1``) can make us easy to access it
by using ``net.mon1``.

.. code-block:: python

    net = nn.Network(syn, lif1, lif2, mon1=mon_lif1, mon2=mon_lif2)

We can simulate the whole network just use ``.run(duration)`` function. Here,
we set the inputs of ``lif1`` object to ``15.``, and open the ``report`` mode.

.. code-block:: python

    net.run(duration=100, inputs=(lif1, 15.), report=True)

Finally, visualize the running results:

.. code-block:: python

    xlim = (0, 100)
    ts = net.run_time()
    fig, gs = nn.visualize.get_figure(n_row=2, n_col=1, len_row=3, len_col=8)
    nn.visualize.plot_potential(net.mon1, ts, ax=fig.add_subplot(gs[0, 0]), xlim=xlim)
    nn.visualize.plot_raster(net.mon1, ts, ax=fig.add_subplot(gs[1, 0]), xlim=xlim, show=True)


It shows

.. figure:: https://github.com/chaoming0625/NumpyBrain/blob/master/docs/images/example.png
    :figclass: align-center
    :width: 500px

Define a Hodgkin–Huxley neuron model
====================================

.. code-block:: python

    import numpy as np
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



Acknowledgements
================

We would like to thank

- Risheng Lian
- Longping Liu

for valuable comments and discussions on the project.

.. _How NumpyBrain works: https://numpybrain.readthedocs.io/en/latest/guides/how_it_works.html


