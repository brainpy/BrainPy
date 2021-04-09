
![Logo](docs/images/logo.png)

[![LICENSE](https://anaconda.org/brainpy/brainpy/badges/license.svg)](https://github.com/PKU-NIP-Lab/BrainPy)    [![Documentation](https://readthedocs.org/projects/brainpy/badge/?version=latest)](https://brainpy.readthedocs.io/en/latest/?badge=latest)     [![Conda](https://anaconda.org/brainpy/brainpy-simulator/badges/version.svg)](https://anaconda.org/brainpy/brainpy-simulator)  [![PyPI version](https://badge.fury.io/py/brainpy-simulator.svg)](https://badge.fury.io/py/brainpy-simulator)



# Why to use BrainPy

``BrainPy`` is an integrative framework for computational neuroscience and brain-inspired computation. Three core functions are provided in BrainPy:

- *General numerical solvers* for ODEs and SDEs (support for DDEs and FDEs will come soon).
- *Neurodynamics simulation tools* for brain objects, such like neurons, synapses and networks (support for soma and dendrites will come soon).
- *Neurodynamics analysis tools* for differential equations, including phase plane analysis and bifurcation analysis (support for continuation analysis and sensitive analysis will come soon).

Moreover, `BrainPy` is designed to effectively satisfy your basic requirements: 

- *Easy to learn and use*, because BrainPy is only based on Python language and has little dependency requirements; 
- *Highly flexible and transparent*, because BrainPy endows the users with the fully data/logic flow control; 
- *Simulation can be guided with the analysis*, because the same code in BrainPy can not only be used for simulation, but also for dynamics analysis; 
- *Efficient running speed*, because BrainPy is compatible with the latest JIT compilers or any other accelerating framework you prefer (below we list the speed comparison based on Numba JIT).


![Speed Comparison](docs/images/speed.png)

`BrainPy` is a backend-independent neural simulator. Users can define models with any backend they prefer. Intrinsically, BrainPy supports the array/tensor-oriented backends such like [NumPy](https://numpy.org/), [PyTorch](https://pytorch.org/), and [TensorFlow](https://www.tensorflow.org/), it also supports the JIT compilers such as [Numba](https://numba.pydata.org/) on CPU or CUDA devices. Extending BrainPy to support other backend frameworks you prefer is very easy. The details please see documents coming soon. 



# Installation

Install ``BrainPy`` by using ``pip``:

```bash
> pip install brainpy-simulator>=1.0.0b1
```

Install ``BrainPy`` by using ``conda``:

```bash
> conda install brainpy-simulator -c brainpy
```

Install ``BrainPy`` from source:

```bash
> pip install git+https://github.com/PKU-NIP-Lab/BrainPy
> # or
> pip install git+https://git.openi.org.cn/OpenI/BrainPy
> # or
> pip install -e git://github.com/PKU-NIP-Lab/BrainPy.git@V0.2.5
```

``BrainPy`` is based on Python (>=3.7), and the following packages are required to be installed to use ``BrainPy``:

- NumPy >= 1.13
- Matplotlib >= 3.3



# Let's start

- **Website (including documentations):** https://brainpy.readthedocs.io/
- **Source code:** https://github.com/PKU-NIP-Lab/BrainPy  or  https://git.openi.org.cn/OpenI/BrainPy
- **Bug reports:** https://github.com/PKU-NIP-Lab/BrainPy/issues  or  Email to adaduo@outlook.com
- **Examples from papers**: https://brainmodels.readthedocs.io/en/latest/from_papers.html

Here list several simple examples for neurodynamics simulation and analysis. Comprehensive examples and tutorials please see [BrainModels](https://brainmodels.readthedocs.io).

<table border="0">
    <tr>
        <td border="0" width="30%">
            <a href="https://github.com/PKU-NIP-Lab/BrainModels/blob/main/brainmodels/tensor_backend/neurons/HodgkinHuxley_model.py">
            <img src="docs/images/HH_neuron.png">
            </a>
        </td>
        <td border="0" valign="top">
            <h3><a href="https://github.com/PKU-NIP-Lab/BrainModels/blob/main/brainmodels/tensor_backend/neurons/HodgkinHuxley_model.py">HH Neuron Model</a></h3>
            <p>The Hodgkin–Huxley neuron model.</p>
        </td>
    </tr>
    <tr>
        <td border="0" width="30%">
            <a href="https://github.com/PKU-NIP-Lab/BrainModels/blob/main/brainmodels/tensor_backend/synapses/AMPA_synapse.py">
            <img src="docs/images/AMPA_model.png">
            </a>
        </td>
        <td border="0" valign="top">
            <h3><a href="https://github.com/PKU-NIP-Lab/BrainModels/blob/main/brainmodels/tensor_backend/synapses/AMPA_synapse.py">AMPA Synapse Model</a></h3>
            <p>The AMPA synapse model.</p>
        </td>
    </tr>
    <tr>
        <td border="0" width="30%">
            <a href="https://brainmodels.readthedocs.io/en/latest/from_papers/Wang_1996_gamma_oscillation.html">
            <img src="docs/images/gamma_oscillation.png">
            </a>
        </td>
        <td border="0" valign="top">
            <h3><a href="https://brainmodels.readthedocs.io/en/latest/from_papers/Wang_1996_gamma_oscillation.html">Gamma Oscillation Model</a></h3>
            <p>Implementation of the paper: <i> Wang, Xiao-Jing, and György Buzsáki. “Gamma oscillation by
                  synaptic inhibition in a hippocampal interneuronal network
                  model.” Journal of neuroscience 16.20 (1996): 6402-6413. </i>
            </p>
        </td>
    </tr>
    <tr>
        <td border="0" width="30%">
            <a href="https://brainmodels.readthedocs.io/en/latest/from_papers/Vreeswijk_1996_EI_net.html">
            <img src="docs/images/EI_balance_net.png">
            </a>
        </td>
        <td border="0" valign="top">
            <h3><a href="https://brainmodels.readthedocs.io/en/latest/from_papers/Vreeswijk_1996_EI_net.html">E/I Balance Network</a></h3>
        <p>Implementation of the paper: <i>Van Vreeswijk, Carl, and Haim Sompolinsky. 
        “Chaos in neuronal networks with balanced excitatory and inhibitory activity.” 
        Science 274.5293 (1996): 1724-1726.</i></p>        
	</td>
    </tr>
    <tr>
        <td border="0" width="30%">
            <a href="https://brainmodels.readthedocs.io/en/latest/from_papers/Wu_2008_CANN.html">
            <img src="docs/images/CANN1d.png">
            </a>
        </td>
        <td border="0" valign="top">
            <h3><a href="https://brainmodels.readthedocs.io/en/latest/from_papers/Wu_2008_CANN.html">Continuous-attractor Network</a></h3>
            <p>Implementation of the paper: <i> Si Wu, Kosuke Hamaguchi, and Shun-ichi Amari. "Dynamics and
                    computation of continuous attractors." Neural
                    computation 20.4 (2008): 994-1025. </i>
            </p>
        </td>
    </tr>
    <tr>
        <td border="0" width="30%">
            <a href="https://brainmodels.readthedocs.io/en/latest/tutorials/dynamics_analysis/NaK_model_analysis.html">
            <img src="docs/images/phase_plane_analysis1.png">
            </a>
        </td>
        <td border="0" valign="top">
            <h3><a href="https://brainmodels.readthedocs.io/en/latest/tutorials/dynamics_analysis/NaK_model_analysis.html">Phase Plane Analysis</a></h3>
            <p>Phase plane analysis of the I<sub>Na,p+</sub>-I<sub>K</sub> model, where
            "input" is 50., and "Vn_half" is -45..</p>
        </td>
    </tr>
    <tr>
        <td border="0" width="30%">
            <a href="https://brainmodels.readthedocs.io/en/latest/tutorials/dynamics_analysis/FitzHugh_Nagumo_analysis.html">
            <img src="docs/images/FitzHugh_Nagumo_codimension1.png">
            </a>
        </td>
        <td border="0" valign="top">
            <h3><a href="https://brainmodels.readthedocs.io/en/latest/tutorials/dynamics_analysis/FitzHugh_Nagumo_analysis.html">
                Codimension 1 Bifurcation Analysis</a></h3>
            <p>Codimension 1 bifurcation analysis of FitzHugh Nagumo model, in which
                "a" is equal to 0.7, and "Iext" is varied in [0., 1.].</p>
        </td>
    </tr>
    <tr>
        <td border="0" width="30%">
            <a href="https://brainmodels.readthedocs.io/en/latest/tutorials/dynamics_analysis/FitzHugh_Nagumo_analysis.html#Codimension-2-bifurcation-analysis">
            <img src="docs/images/FitzHugh_Nagumo_codimension2.png">
            </a>
        </td>
        <td border="0" valign="top">
            <h3><a href="https://brainmodels.readthedocs.io/en/latest/tutorials/dynamics_analysis/FitzHugh_Nagumo_analysis.html#Codimension-2-bifurcation-analysis">
                Codimension 2 Bifurcation Analysis</a></h3>
            <p>Codimension 2 bifurcation analysis of FitzHugh Nagumo model, in which "a"
               is varied in [0.5, 1.0], and "Iext" is varied in [0., 1.].</p>
        </td>
    </tr>
</table>


