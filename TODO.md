[TOC]









# Version 1.1.0-alpha



## Base

- [x] ``Collector``: must check whether the ``(key,value)`` pair are duplicate  (done @2021/08/26 by @chaoming)
- [x] ``nodes``: slove the problem of circular reference (done @2021/08/24 by @chaoming)
- [x] ``ints``: get integrators based on all nodes (done @2021/08/24 by @chaoming)
- [x] Reimplement ``nodes`` and ``ints`` in each children class, like Module, Sequential, DynamicaSystem, Container, Network (done @2021/08/24 by @chaoming)
- [x] ``ints``: "subsets()" method can get a subset of integrator, like ``ODE_INT``, ``SDE_INT``, etc. (done @2021/08/24 by @chaoming)





## Math

- [x] **[Numpy]:** support JIT compilation in numpy backend (done @ 2021.09.01 by @chaoming)
- [x] **[Numpy]:** support 'fft' (done @ 2021.09.03 by @chaoming)
- [ ] support to set `dt`  in the single object level (i.e., single instance of DynamicSystem)
- [ ] **[JAX]:** "random" module 
- [ ] **[JAX]:** math operations
- [ ] **[JAX]:** vmap
- [ ] **[JAX]:** pmap
- [ ] **[JAX]:** control conditions: 
- [x] **[JAX]:** support 'fft' (done @ 2021.09.03 by @chaoming)
- [x] **[JAX]:** **IMPORTANT!!!** Change API of `grad()` and `value_and_grad()`: There are bugs in the gradient functions. Gradient computation also needs to inspect the variable types. Moreover, it is independent from the JIT function. Therefore, we should pass dynamical variables into the gradient functions too. (done @ 2021.08.26 by @chaoming)
- [x] **[JAX]:** **IMPORTANT!!!** change API of `vars()`: we should refer Dynamical Variables as `Variable`; We can not retrieve every "JaxArray" from `vars()`, otherwise the whole system will waste a lot of time on useless assignments. (done @ 2021.08.25 by @chaoming)
- [x] **[JAX]:** change API of `brainpy.math.jit(target)`, please return another class if `target` is not a function [abolished]
- [x] **[JAX]:** ``JaxArray`` Wrapper for JAX `ndarray`
  - [x] register pytree  (done @ 2021.06.15 by @chaoming)
  - [x] support `ndarray` intrinsic methods: 
    - [x] functions in NumPy ndarray: any(), all() .... view() (done @ 2021.06.30 by @chaoming)
    - [ ] functions in JAX DeviceArray: 
  - [x] numpy methods in JaxArray (done @ 2021.08.25, @2021.08.28 by @chaoming)
  - [x] documentation for JaxArray methods (done @ 2021.08.25 by @chaoming)
  - [ ] test for ndarray wrapper 
- [x] **[JAX]:** Support JIT in JAX (done @ 2021.07.30 by @chaoming)
- [x] **[JAX]:** support gradient ``grad()`` in JAX (done @ 2021.07.30 by @chaoming)







## Integrators

- [ ] FDEs
  - [ ] Support numerical integration for fractional differential equations (FDEs)
- [ ] DDEs
  - [ ] Support numerical integration for delayed differential equations (DDEs)
- [ ] SDEs
  - [ ] More convenient way to define constant Wiener term in SDEs
  - [ ] Check whether the user-defined code has the keywords (like `f`, `g`, etc.) in SDEs 
- [x] doc comments for ODE APIs (done @2021/08/28 by @chaoming)
- [x] The unique name of the ODE, SDE, DDE, FDE integrators (done @2021/08/23 by @chaoming)




## Simulation

- [ ] Allow defining the `Soma` object
- [ ] Allow defining the `Dendrite` object





## Analysis

- [ ] Support numerical continuation for ODEs
- [x] Analyze the parameter by self (done @2021/09/02 by @chaoming)





## DNN

- [ ] "objectives" module: commonly used loss functions 
- [x] 'dnn' module documentation in 'index.rst'   (done @2021/08/26 by @chaoming)








## Documentation

- [ ] detailed documentation for numerical solvers of SDEs
- [ ] doc comments for ODEs, like Euler, RK2, etc. We should provide the detailed mathematical equations, and the corresponding suggestions for the corresponding algorithm. 
- [x] APIs for integrators  (done @2021/08/23 by @chaoming)
- [x] installation instruction, especially package dependency  (done @2021/08/23 by @chaoming)





## Examples

- [ ] DNNs: Deep Neural Networks
- [ ] RNNs: Learning recurrent networks by back-propagation
- [ ] RNNs: Learning recurrent networks by biological plausible algorithms
- [ ] Network Models: working memory (Misha, 2008)
- [x] network example: decision making (Xiaojing Wang, Neuron, 2002) (done @2021/08/22 by @chaoming)





## Others

- [ ] publish `BrainPy` on `"conda-forge"`: https://conda-forge.org/docs/maintainer/adding_pkgs.html#





# Version 1.0.2



## Numerical Solvers

- [x] ODEs
  - [x] Check whether the user-defined code has the keywords (like `f`, etc.) in ODEs  (done @ 2021.05.29)




## Dynamics Simulation

- [x] name
  - [x] check unique `name` for each DynamicSystem instance (done @ 2021.06.30)
- [x] vars(), nodes(), ints()
  - [x] relative access, relative path  (done @ 2021.06.30)
  - [x] absolute access, absolute path (done @ 2021.06.30)
- [x] Monitor
  - [x] Allow running monitor functions by customized `@every` xx ms (done @ 2021.05.23)
  - [x] Monitor support for the multi-dimensional variables in numpy (done @ 2021.05.23)
  - [x] Monitor support for the multi-dimensional variables in numba (done @ 2021.05.23)
  - [x] Monitor index support in numpy (done @ 2021.05.23)
  - [x] Monitor index support in numba (done @ 2021.05.23)
  - [x] Monitor interval support in numpy (done @ 2021.05.23)
  - [x] Monitor interval support in numba (done @ 2021.05.23)
- [x] Running order
  - [x] Allow running step functions by customized `@every` xx ms in numpy (done @ 2021.05.22)
  - [x] Allow running step functions by customized `@every` xx ms in numba (done @ 2021.05.22)
  - [x] Allow customizing the running order schedule in numpy (for both object and network) (done @ 2021.05.22)
  - [x] Allow customizing the running order schedule in numba (for both object and network) (done @ 2021.05.22)
  - [x] Allow customizing the running order schedule in numba-cuda (for both object and network) (done @ 2021.05.22)





## Backend & Drivers

- [x] relax `targe_backend` setting. Default is `general`, if not setting. (done @ 2021.06.30)
- [x] move backend setting into the `brainpy.math` module (done @ 2021.06.30)
- [x] move global `dt` setting into the `brainpy.math` module (done @ 2021.06.30)
- [x] support global `dt` setting




## Documentation

- [x] detailed documentation for numerical solvers of ODEs (done @ 2021.05.29)
- [x] unified operation (done @ 2021.05.26)
- [x] more about monitor (done @ 2021.05.25)
- [x] repeat running mode  (done @ 2021.05.25)
- [x] running order scheduling  (done @ 2021.05.25)



