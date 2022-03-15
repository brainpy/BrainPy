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
    <a href="https://github.com/PKU-NIP-Lab/BrainPy"><img alt="Linux CI" src="https://github.com/PKU-NIP-Lab/BrainPy/actions/workflows/MacOS_CI.yml/badge.svg"></a>
</p>




BrainPy is a flexible, efficient, and extensible framework for computational neuroscience and brain-inspired computation based on the Just-In-Time (JIT) compilation (built on top of [JAX](https://github.com/google/jax)). It provides an integrative ecosystem for brain dynamics programming, including brain dynamics **simulation**, **training**, **analysis**, etc. 

- **Website (documentation and APIs)**: https://brainpy.readthedocs.io/en/latest
- **Source**: https://github.com/PKU-NIP-Lab/BrainPy
- **Bug reports**: https://github.com/PKU-NIP-Lab/BrainPy/issues
- **Source on OpenI**: https://git.openi.org.cn/OpenI/BrainPy
- **Canonical brain models**: https://brainmodels.readthedocs.io/
- **Examples from literature**: https://brainpy-examples.readthedocs.io/



## Installation

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



### 1. Operator level

Mathematical operators in BrainPy are the same as those in NumPy. 

```python
>>> import numpy as np
>>> import brainpy.math as bm

# array creation
>>> np_arr = np.zeros((2, 4));  np_arr  
array([[0., 0., 0., 0.],
       [0., 0., 0., 0.]])
>>> bm_arr = bm.zeros((2, 4));  bm_arr
JaxArray([[0., 0., 0., 0.],
          [0., 0., 0., 0.]], dtype=float32)

# in-place updating
>>> np_arr[0] += 1.;  np_arr
array([[1., 1., 1., 1.],
       [0., 0., 0., 0.]])
>>> bm_arr[0] += 1.;  bm_arr
JaxArray([[1., 1., 1., 1.],
          [0., 0., 0., 0.]], dtype=float32)

# mathematical functions
>>> np.sin(np_arr)
array([[0.84147098, 0.84147098, 0.84147098, 0.84147098],
       [0.        , 0.        , 0.        , 0.        ]])
>>> bm.sin(bm_arr)
JaxArray([[0.84147096, 0.84147096, 0.84147096, 0.84147096],
          [0.        , 0.        , 0.        , 0.        ]],  dtype=float32)

# linear algebra
>>> np.dot(np_arr, np.ones((4, 2)))
array([[4., 4.],
       [0., 0.]])
>>> bm.dot(bm_arr, bm.ones((4, 2)))
JaxArray([[4., 4.],
          [0., 0.]], dtype=float32)

# random number generation
>>> np.random.uniform(-0.1, 0.1, (2, 3))
array([[-0.02773637,  0.03766689, -0.01363128],
       [-0.01946991, -0.06669802,  0.09426067]])
>>> bm.random.uniform(-0.1, 0.1, (2, 3))
JaxArray([[-0.03044081, -0.07787752,  0.04346445],
          [-0.01366713, -0.0522548 ,  0.04372055]], dtype=float32)
```



### 2. Integrator level

Numerical methods for ordinary differential equations (ODEs). 

```python
sigma = 10; beta = 8/3; rho = 28

@bp.odeint(method='rk4')
def lorenz_system(x, y, z, t):
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return dx, dy, dz

runner = bp.integrators.IntegratorRunner(lorenz_system, dt=0.01)
runner.run(100.)
```



Numerical methods for stochastic differential equations (SDEs). 

```python
sigma = 10; beta = 8/3; rho = 28
p=0.1

def lorenz_noise(x, y, z, t):
    return p*x, p*y, p*z

@bp.odeint(method='milstein', g=lorenz_noise)
def lorenz_system(x, y, z, t):
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return dx, dy, dz

runner = bp.integrators.IntegratorRunner(lorenz_system, dt=0.01)
runner.run(100.)
```



Numerical methods for delay differential equations (SDEs). 

```python
xdelay = bm.FixedLenDelay(1, delay_len=1., before_t0=1., dt=0.01)

@bp.ddeint(method='rk4', state_delays={'x': xdelay})
def second_order_eq(x, y, t):
    dx = y
    dy = -y - 2*x - 0.5*xdelay(t-1)
    return dx, dy

runner = bp.integrators.IntegratorRunner(second_order_eq, dt=0.01)
runner.run(100.)
```



### 3. Dynamics simulation level

Building an E-I balance network.

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
    

net = EINet()
runner = bp.dyn.DSRunner(net)
runner(100.)
```



### 4. Dynamics training level

Training an  echo state network.

```python
i = bp.nn.Input(3)
r = bp.nn.Reservoir(100)
o = bp.nn.LinearReadout(3)

net = i >> r >> o

trainer = bp.nn.RidgeTrainer(net, beta=1e-5) # Ridge Regression

trainer = bp.nn.FORCELearning(net, alpha=1.) # FORCE Learning
```



Training a next-generation reservoir computing model.

```python
i = bp.nn.Input(3)
r = bp.nn.NVAR(delay=2, order=2)
o = bp.nn.LinearReadout(3)

net = i >> r >> o

trainer = bp.nn.RidgeTrainer(net, beta=1e-5)
```



Training an artificial recurrent neural network. 

```python
i = bp.nn.Input(3)
l1 = bp.nn.VanillaRNN(100)
l2 = bp.nn.VanillaRNN(200)
o = bp.nn.Dense(10)

net = i >> l1 >> l2 >> o

trainer = bp.nn.BPTT(net, 
                     loss='cross_entropy_loss',
                     optimizer=bp.optim.Adam(0.01))
```



### 5. Dynamics analysis level

Analyzing a low-dimensional FitzHugh–Nagumo neuron model.

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


### 6. More others

For **more functions and examples**, please refer to the [documentation](https://brainpy.readthedocs.io/) and [examples](https://brainpy-examples.readthedocs.io/).




## License

[GNU General Public License v3.0](https://github.com/PKU-NIP-Lab/BrainPy/blob/master/LICENSE)
