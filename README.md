<p align="center">
  	<img alt="Header image of BrainPy - brain dynamics programming in Python." src="./images/logo.png" width=80%>
</p> 



<p align="center">
	<a href="https://pypi.org/project/brain-py/"><img alt="Supported Python Version" src="https://img.shields.io/pypi/pyversions/brain-py"></a>
	<a href="https://github.com/PKU-NIP-Lab/BrainPy"><img alt="LICENSE" src="https://anaconda.org/brainpy/brainpy/badges/license.svg"></a>
  	<a href="https://brainpy.readthedocs.io/en/latest/?badge=latest"><img alt="Documentation" src="https://readthedocs.org/projects/brainpy/badge/?version=latest"></a>
  	<a href="https://badge.fury.io/py/brain-py"><img alt="PyPI version" src="https://badge.fury.io/py/brain-py.svg"></a>
    <a href="https://github.com/PKU-NIP-Lab/BrainPy"><img alt="Linux CI" src="https://github.com/PKU-NIP-Lab/BrainPy/actions/workflows/Linux_CI.yml/badge.svg"></a>
    <a href="https://github.com/PKU-NIP-Lab/BrainPy"><img alt="Linux CI" src="https://github.com/PKU-NIP-Lab/BrainPy/actions/workflows/Windows_CI.yml/badge.svg"></a>
</p>




BrainPy is a flexible, efficient, and extensible framework for computational neuroscience and brain-inspired computation based on the Just-In-Time (JIT) compilation (built on top of [JAX](https://github.com/google/jax)). It provides an integrative ecosystem for brain dynamics programming, including brain dynamics **simulation**, **training**, **analysis**, etc. 

- **Website (documentation and APIs)**: https://brainpy.readthedocs.io/en/latest
- **Source**: https://github.com/PKU-NIP-Lab/BrainPy
- **Bug reports**: https://github.com/PKU-NIP-Lab/BrainPy/issues
- **Source on OpenI**: https://git.openi.org.cn/OpenI/BrainPy
- **Canonical brain models**: https://brainmodels.readthedocs.io/
- **Examples from literature**: https://brainpy-examples.readthedocs.io/



## Install

BrainPy is based on Python (>=3.6) and can be installed on  Linux (Ubuntu 16.04 or later), macOS (10.12 or later), and Windows platforms. Install the latest version of BrainPy:

```bash
$ pip install brain-py -U
```

The following packages are required for ``BrainPy``: 

`numpy >= 1.15` and `jax >= 0.2.10` ([how to install jax?](https://brainpy.readthedocs.io/en/latest/quickstart/installation.html#dependency-2-jax))

For detailed installation instructions, please refer to the documentation: [Quickstart/Installation](https://brainpy.readthedocs.io/en/latest/quickstart/installation.html)



## Examples



```python
import brainpy as bp
```



**1\. E-I balance network**

```python
class EINet(bp.dyn.Network):
  def __init__(self):
    E = bp.dyn.LIF(3200, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.)
	I = bp.dyn.LIF(800, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.)
    E.V[:] = bp.math.random.randn(3200) * 2 - 60.
    I.V[:] = bp.math.random.randn(800) * 2 - 60.
        
    E2E = bp.dyn.ExpCOBA(E, E, bp.conn.FixedProb(prob=0.02), E=0., g_max=0.6, tau=5.)
    E2I = bp.dyn.ExpCOBA(E, I, bp.conn.FixedProb(prob=0.02), E=0., g_max=0.6, tau=5.)
    I2E = bp.dyn.ExpCOBA(I, E, bp.conn.FixedProb(prob=0.02), E=-80., g_max=6.7, tau=10.)
    I2I = bp.dyn.ExpCOBA(I, I, bp.conn.FixedProb(prob=0.02), E=-80., g_max=6.7, tau=10.)
        
    super(EINet, self).__init__(E2E, E2I, I2E, I2I, E=E, I=I)
```



**2\. Echo state network**

```python
net = (
    bp.nn.Input(3)
    >>
    bp.nn.Reservoir(100)
    >> 
    bp.nn.LinearReadout(3)
)

# Ridge Regression
trainer = bp.nn.RidgeTrainer(net, beta=1e-5)

# FORCE Learning
trainer = bp.nn.FORCELearning(net, alpha=1.)
```



**3. Next generation reservoir computing**

```python
net = (
    bp.nn.Input(3)
    >>
    bp.nn.NVAR(delay=2, order=2)
    >> 
    bp.nn.LinearReadout(3)
)

trainer = bp.nn.RidgeTrainer(net, beta=1e-5)
```



**4. Recurrent neural network**

```python
net = (
    bp.nn.Input(10)
    >>
    bp.nn.VanillaRNN(100)
    >> 
    bp.nn.VanillaRNN(200)
    >>
    bp.nn.Dense(10)
)

trainer = bp.nn.BPTT(net, 
                     loss='cross_entropy_loss',
                     optimizer=bp.optim.Adam(0.01))
```



**5\. Analyzing a low-dimensional FitzHugh–Nagumo neuron model**

```python
bp.math.enable_x64()

model = bp.dyn.FHN(1)
analyzer = bp.analysis.PhasePlane2D(model,
                                    target_vars={'V': [-3, 3], 'w': [-3., 3.]},
                                    pars_update={'I_ext': 0.8}, 
                                    resolutions=0.01)
analyzer.plot_nullcline()
analyzer.plot_vector_field()
analyzer.plot_fixed_point()
analyzer.plot_trajectory({'V': [-2.8], 'w': [-1.8]}, duration=100.)
analyzer.show_figure()
```

<p align="center"><img src="./docs/_static/fhn_ppa.png" width="60%">
</p> 



## License

[GNU General Public License v3.0](https://github.com/PKU-NIP-Lab/BrainPy/blob/master/LICENSE)
