# Release notes (``brainpylib``)

## Version 0.3.0

- Support `brainpy>=2.5.0`
- Fix bugs on windows platform
- remove all customized C++ and CUDA operators

## Version 0.2.8

- Support `brainpy>=2.5.0`
- Fix bugs that the DLL cannot be loaded correctly when windows does not have a c++ environment,

## ~~Version 0.2.7(YANKED)~~

## Version 0.2.6

- Support `brainpy>=2.5.0`
- Fix bugs of taichi call function for single result

## Version 0.2.5

- Add new taichi call function for single result on CPU backend

## Version 0.2.4

- Add taichi customized operator call on arm64 backend

## ~~Version 0.2.3(YANKED)~~

## Version 0.2.2

- Fix bugs of just-in-time connectivity operators on CPU device

## Version 0.2.1

- Fix bugs of Taichi AOT call on GPU backend by ``cudaMemset()`` CUDA arrays

## Version 0.2.0

- Add XLA custom call from [Taichi](https://github.com/taichi-dev/taichi) AOT (ahead of time) operators on both CPU and
  GPU platforms

## Version 0.0.5

- Support operator customization on GPU by ``numba``

## Version 0.0.4

- Support operator customization on CPU by ``numba``

## Version 0.0.3

- Support ``event_sum()`` operator on GPU
- Support ``event_prod()`` operator on CPU
- Support ``atomic_sum()`` operator on GPU
- Support ``atomic_prod()`` operator on CPU and GPU

## Version 0.0.2

- Support ``event_sum()`` operator on CPU
- Support ``event_sum2()`` operator on CPU
- Support ``atomic_sum()`` operator on CPU

