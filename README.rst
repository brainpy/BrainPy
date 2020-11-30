

.. image:: docs/images/logo.png
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
The goal of ``BrainPy`` is to provide a unified simulation and analysis framework
for neuronal dynamics with the feature of high flexibility and efficiency.
BrainPy is flexible because it endows the users with the fully data/logic flow control.
BrainPy is efficient because it supports JIT acceleration on CPUs
(see the following comparison figure. In future, we will support JIT acceleration on GPUs).

.. figure:: https://github.com/PKU-NIP-Lab/NumpyBrain/blob/master/docs/images/speed.png
    :alt: Speed of BrainPy
    :figclass: align-center
    :width: 250px


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

- Numba >= 0.50.0
- TensorFlow >= 2.4
- PyTorch >= 1.7


Neurodynamics simulation
========================


.. raw:: html

    <table border="0">
        <tr>
            <td border="0" width="30%">
                <a href="examples/neurons/HH_model.py">
                <img src="docs/images/HH_neuron.png">
                </a>
            </td>
            <td border="0" valign="top">
                <h3><a href="examples/neurons/HH_model.py">HH Neuron Model</a></h3>
                <p>The Hodgkin–Huxley model, or conductance-based model,
                is a mathematical model that describes how action potentials
                in neurons are initiated and propagated. It is a set of nonlinear
                differential equations that approximates the electrical characteristics
                of excitable cells such as neurons and cardiac myocytes.</p>
            </td>
        </tr>
        <tr>
            <td border="0" width="30%">
                <a href="examples/synapses/AMPA_vector.py">
                <img src="docs/images/AMPA_model.png">
                </a>
            </td>
            <td border="0" valign="top">
                <h3><a href="examples/synapses/AMPA_vector.py">AMPA Synapse Model</a></h3>
                <p>AMPA synapse model.</p>
            </td>
        </tr>
        <tr>
            <td border="0" width="30%">
                <a href="examples/networks/gamma_oscillation.py">
                <img src="docs/images/gamma_oscillation.png">
                </a>
            </td>
            <td border="0" valign="top">
                <h3><a href="examples/networks/gamma_oscillation.py">Gamma Oscillation Model</a></h3>
                <p>Implementation of the paper: <i> Wang, Xiao-Jing, and György Buzsáki. “Gamma oscillation by
                      synaptic inhibition in a hippocampal interneuronal network
                      model.” Journal of neuroscience 16.20 (1996): 6402-6413. </i>
                </p>
            </td>
        </tr>
        <tr>
            <td border="0" width="30%">
                <a href="examples/networks/EI_balance_network.py">
                <img src="docs/images/EI_balance_net.png">
                </a>
            </td>
            <td border="0" valign="top">
                <h3><a href="examples/networks/EI_balance_network.py">E/I Balance Network</a></h3>
            </td>
        </tr>
        <tr>
            <td border="0" width="30%">
                <a href="examples/networks/CANN_1D.py">
                <img src="docs/images/CANN1d.png">
                </a>
            </td>
            <td border="0" valign="top">
                <h3><a href="examples/networks/CANN_1D.py">Continuous-attractor Network</a></h3>
                <p>Implementation of the paper: <i> Si Wu, Kosuke Hamaguchi, and Shun-ichi Amari. "Dynamics and
                        computation of continuous attractors." Neural
                        computation 20.4 (2008): 994-1025. </i>
                </p>
            </td>
        </tr>
    </table>


More neuron examples please see `examples/neurons <https://github.com/PKU-NIP-Lab/BrainPy/tree/master/examples/neurons>`_.

More synapse examples please see `examples/synapses <https://github.com/PKU-NIP-Lab/BrainPy/tree/master/examples/synapses>`_.

More network examples please see `examples/networks <https://github.com/PKU-NIP-Lab/BrainPy/tree/master/examples/networks>`_.


Neurodynamics analysis
======================


.. raw:: html

    <table border="0">
        <tr>
            <td border="0" width="30%">
                <a href="examples/dynamics_analysis/phase_portrait_of_NaK_model.py">
                <img src="docs/images/phase_plane_analysis1.png">
                </a>
            </td>
            <td border="0" valign="top">
                <h3><a href="examples/dynamics_analysis/phase_portrait_of_NaK_model.py">Phase Plane Analysis</a></h3>
                <p>Phase plane analysis of a two-dimensional system model:
                    the I<sub>Na,p+</sub>-I<sub>K</sub> model.</p>
            </td>
        </tr>
        <tr>
            <td border="0" width="30%">
                <a href="examples/dynamics_analysis/1D_system_bifur_codim1.py">
                <img src="docs/images/codimension1.png">
                </a>
            </td>
            <td border="0" valign="top">
                <h3><a href="examples/dynamics_analysis/1D_system_bifur_codim1.py">Codimension 1 Bifurcation Analysis</a></h3>
                <p>Codimension 1 bifurcation analysis of a one-dimensional system.</p>
            </td>
        </tr>
        <tr>
            <td border="0" width="30%">
                <a href="examples/dynamics_analysis/2D_system_bifur_codim2.py">
                <img src="docs/images/codimension2.png">
                </a>
            </td>
            <td border="0" valign="top">
                <h3><a href="examples/dynamics_analysis/2D_system_bifur_codim2.py">
                    Codimension 2 Bifurcation Analysis</a></h3>
                <p>Codimension 2 bifurcation analysis of a two-variable neuron model:
                    the I<sub>Na,p+</sub>-I<sub>K</sub> model.</p>
            </td>
        </tr>
        <tr>
            <td border="0" width="30%">
                <a href="examples/dynamics_analysis/FitzHugh_Nagumo_analysis.py">
                <img src="docs/images/FitzHugh_Nagumo_codimension2.png">
                </a>
            </td>
            <td border="0" valign="top">
                <h3><a href="examples/dynamics_analysis/FitzHugh_Nagumo_analysis.py">
                    Codimension 2 Bifurcation Analysis 2</a></h3>
                <p>Codimension 2 bifurcation analysis of FitzHugh Nagumo model.</p>
            </td>
        </tr>
    </table>


More examples please see
`examples/dynamics_analysis <https://github.com/PKU-NIP-Lab/BrainPy/tree/master/examples/dynamics_analysis>`_.


