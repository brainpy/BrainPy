
Usage of ``connect`` module
============================

Here, NumpyBrain pre-defined several commonly used connection methods.

.. contents::
    :local:
    :depth: 2

one-to-one connection
---------------------

The neurons in the pre-synaptic neuron group only connect to the neurons
in the same position of the post-synaptic group. Thus, this connection
requires the geometry of two neuron groups same. Otherwise, an error will
occurs.

.. figure:: ../images/one2one.png
    :alt: one2one connection
    :width: 200px
    :figclass: align-center

Usage of the method:

.. code-block:: python

    import npbrain as nn

    pre_ids, post_ids, anchors = nn.connect.one2one(num_pre, num_post)

all-to-all connection
---------------------

All neurons of the post-synaptic population form connections with all
neurons of the pre-synaptic population (dense connectivity). Users can
choose whether connect the neurons at the same position
(`include_self=True or False`).

.. figure:: ../images/all2all.png
    :alt: all2all connection
    :width: 200px
    :figclass: align-center

Usage of the method:

.. code-block:: python

    pre_ids, post_ids, anchors = nn.connect.one2one(num_pre, num_post, include_self=True)


grid-four connection
--------------------

`grid-four connection` is four nearest neighbors connection. Each neuron connect to its
nearest four neurons.

.. figure:: ../images/grid_four.png
    :alt: grid_four connection
    :width: 250px
    :figclass: align-center

.. code-block:: python

    pre_ids, post_ids, anchors = nn.connect.grid_four(height, width, include_self=True)


grid-eight connection
---------------------

`grid-eight connection` is eight nearest neighbors connection. Each neuron connect to its
nearest eight neurons.

.. figure:: ../images/grid_eight.png
    :alt: grid_eight connection
    :width: 250px
    :figclass: align-center

.. code-block:: python

    pre_ids, post_ids, anchors = nn.connect.grid_eight(height, width, include_self=True)



grid-N connection
-----------------


`grid-N connection` is also a nearest neighbors connection. Each neuron connect to its
nearest :math:`2N \cdot 2N` neurons.

.. figure:: ../images/grid_N.png
    :alt: grid_N connection
    :width: 250px
    :figclass: align-center


.. code-block:: python

    pre_ids, post_ids, anchors = nn.connect.grid_N(height, width, N=1, include_self=True)



fixed_probability connection
----------------------------

For each post-synaptic neuron, there is a fixed probability that it forms a connection
with a neuron of the pre-synaptic population. It is basically a all_to_all projection,
except some synapses are not created, making the projection sparser.

.. figure:: ../images/fixed_proab.png
    :alt: fixed_proab connection
    :width: 200px
    :figclass: align-center

.. code-block:: python

    pre_ids, post_ids, anchors = nn.connect.fixed_prob(
            num_pre, num_post, prob=0.2, include_self=True, seed=None)


fixed pre-synaptic number connection
------------------------------------

Each neuron in the post-synaptic population receives connections from a
fixed number of neurons of the pre-synaptic population chosen randomly.
It may happen that two post-synaptic neurons are connected to the same
pre-synaptic neuron and that some pre-synaptic neurons are connected to
nothing.

.. figure:: ../images/fixed_pre_num.png
    :alt: fixed_pre_num connection
    :width: 200px
    :figclass: align-center

.. code-block:: python

    pre_ids, post_ids, anchors = nn.connect.fixed_prenum(
            num_pre, num_post, num=10, include_self=True, seed=None)



fixed post-synaptic number connection
-------------------------------------

Each neuron in the pre-synaptic population sends a connection to a fixed number of neurons
of the post-synaptic population chosen randomly. It may happen that two pre-synaptic neurons
are connected to the same post-synaptic neuron and that some post-synaptic neurons receive
no connection at all.

.. figure:: ../images/fixed_post_num.png
    :alt: fixed_post_num connection
    :width: 200px
    :figclass: align-center

.. code-block:: python

    pre_ids, post_ids, anchors = nn.connect.fixed_postnum(
            num_pre, num_post, num=10, include_self=True, seed=None)


gaussian probability connection
-------------------------------

Builds a Gaussian connection pattern between the two populations, where
the connection probability decay according to the gaussian function.

Specifically,

.. math::

    p=\exp(-\frac{(x-x_c)^2+(y-y_c)^2}{2\sigma^2})

where :math:`(x, y)` is the position of the pre-synaptic neuron
and :math:`(x_c,y_c)` is the position of the post-synaptic neuron.

For example, in a :math:`30 \textrm{x} 30` two-dimensional networks, when
:math:`\beta = \frac{1}{2\sigma^2} = 0.1`, the connection pattern is shown
as the follows:

.. code-block:: python

    pre_ids, post_ids, anchors = nn.connect.gaussian_prob(
            pre_geometry, post_geometry, sigma=2.236, normalize=False,
            include_self=True, seed=None)

.. figure:: ../images/gaussian_prob.png
    :alt: gaussian_probability connection
    :width: 500px
    :figclass: align-center

gaussian weight connection
--------------------------

Builds a Gaussian connection pattern between the two populations, where
the weights decay with gaussian function.

Specifically,

.. math::

    w(x, y) = w_{max} \cdot \exp(-\frac{(x-x_c)^2+(y-y_c)^2}{2\sigma^2})

where :math:`(x, y)` is the position of the pre-synaptic neuron (normalized
to [0,1]) and :math:`(x_c,y_c)` is the position of the post-synaptic neuron
(normalized to [0,1]), :math:`w_{max}` is the maximum weight. In order to void
creating useless synapses, :math:`w_{min}` can be set to restrict the creation
of synapses to the cases where the value of the weight would be superior
to :math:`w_{min}`. Default is :math:`0.01 w_{max}`.


.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    def show_weight(pre_ids, post_ids, weights, geometry, neu_id):
        height, width = geometry
        ids = np.where(pre_ids == neu_id)[0]
        post_ids = post_ids[ids]
        weights = weights[ids]

        X, Y = np.arange(height), np.arange(width)
        X, Y = np.meshgrid(X, Y)
        Z = np.zeros(geometry)
        for id_, weight in zip(post_ids, weights):
            h, w = id_ // width, id_ % width
            Z[h, w] = weight

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


.. code-block:: python

    pre_ids, post_ids, anchors, weights = nn.connect.gaussian_weight(
        (30, 30), (30, 30), sigma=0.1, w_max=1., w_min=0.,
        normalize=True, include_self=True)

    show_weight(pre_ids, post_ids, weights, (30, 30), 465)


.. figure:: ../images/gaussian_weight.png
    :alt: gaussian_weight connection
    :width: 500px
    :figclass: align-center


difference-of-gaussian (dog) connection
----------------------------------------

Builds a Difference-Of-Gaussian (dog) connection pattern between the two populations.

Mathematically,

.. math::

    w(x, y) = &w_{max}^+ \cdot \exp(-\frac{(x-x_c)^2+(y-y_c)^2}{2\sigma_+^2}) \\
    &- w_{max}^- \cdot \exp(-\frac{(x-x_c)^2+(y-y_c)^2}{2\sigma_-^2})

where weights smaller than :math:`0.01 * abs(w_{max} - w_{min})` are not created and
self-connections are avoided by default (parameter allow_self_connections).


.. code-block:: python

    pre_ids, post_ids, anchors, weights = nn.connect.dog(
        (40, 40), (40, 40), sigmas=[0.08, 0.15], ws_max=[1.0, 0.7], w_min=0.01,
        normalize=True, include_self=True)

    show_weight(pre_ids, post_ids, weights, (40, 40), 820)


.. figure:: ../images/dog.png
    :alt: dog connection
    :width: 500px
    :figclass: align-center
