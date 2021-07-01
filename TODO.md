

# Numerical Solvers

- [ ] Support numerical integration for fractional differential equations (FDEs)
- [ ] Support numerical integration for delayed differential equations (DDEs)
- [ ] More convenient way to define constant Wiener term in SDEs
- [ ] Check whether the user-defined code has the keywords (like `f`, `g`, etc.) in SDEs 
- [x] Check whether the user-defined code has the keywords (like `f`, etc.) in ODEs  (done @ 2021.05.29)




# Neuronal Dynamics Simulation

- [ ] Allow defining the `Soma` object
- [ ] Allow defining the `Dendrite` object
- [ ] name
  - [x] check unique `name` for each DynamicSystem instance (done @ 2021.06.30)
- [x] vars(), nodes(), ints()
  - [x] relative access, relative path  (done @ 2021.06.30)
  - [x] absolute access, absolute path (done @ 2021.06.30)
- [ ] Monitor
  - [x] Allow running monitor functions by customized `@every` xx ms (done @ 2021.05.23)
  - [x] Monitor support for the multi-dimensional variables in numpy (done @ 2021.05.23)
  - [x] Monitor support for the multi-dimensional variables in numba (done @ 2021.05.23)
  - [x] Monitor index support in numpy (done @ 2021.05.23)
  - [x] Monitor index support in numba (done @ 2021.05.23)
  - [x] Monitor interval support in numpy (done @ 2021.05.23)
  - [x] Monitor interval support in numba (done @ 2021.05.23)
- [ ] Running order
  - [x] Allow running step functions by customized `@every` xx ms in numpy (done @ 2021.05.22)
  - [x] Allow running step functions by customized `@every` xx ms in numba (done @ 2021.05.22)
  - [x] Allow customizing the running order schedule in numpy (for both object and network) (done @ 2021.05.22)
  - [x] Allow customizing the running order schedule in numba (for both object and network) (done @ 2021.05.22)
  - [x] Allow customizing the running order schedule in numba-cuda (for both object and network) (done @ 2021.05.22)



# Neuronal Dynamics Analysis

- [ ] Support numerical continuation for ODEs



# Backend & Drivers



- [ ] Wrapper for JAX `ndarray`
  - [x] register pytree  (done @ 2021.06.15)
  - [x] support `ndarray` intrinsic methods: like 
    - [x] any(), all() .... view() (done @ 2021.06.30)
  - [ ] test for ndarray wrapper 
- [ ] Support JIT in JAX
  - [ ] multi-scaling modeling
- [ ] Support Numba
  - [ ] recompile the numba JIT compilation
- [ ] support "buffer" in `brainpy.math` module
- [x] relax `targe_backend` setting. Default is `general`, if not setting. (done @ 2021.06.30)
- [x] move backend setting into the `brainpy.math` module (done @ 2021.06.30)
- [x] move global `dt` setting into the `brainpy.math` module (done @ 2021.06.30)
- [ ] support to set `dt`  in the single object level (i.e., single instance of DynamicSystem)
- [x] support global `dt` setting




# Documentation

- [ ] detailed documentation for numerical solvers of SDEs
- [x] detailed documentation for numerical solvers of ODEs (done @ 2021.05.29)
- [x] unified operation (done @ 2021.05.26)
- [x] more about monitor (done @ 2021.05.25)
- [x] repeat running mode  (done @ 2021.05.25)
- [x] running order scheduling  (done @ 2021.05.25)

