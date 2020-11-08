Tips on JIT programming
=======================

BrainPy heavily relies on JIT compilation. Minimal knowledge about
JIT programming will make you code much faster codes on BrainPy.


Global variable cannot be updated
-----------------------------------

JIT compilers treat global variables as compile-time constants, which
means during the updating of step functions, global variables will not 
be updated. Anything you want to modify must be passed as an argument 
in the step functions. For example, if you define a 1D array, and at
every call of the step function, some value want to be stored in the array:

.. code-block:: python

    import brainpy as bp
    import brainpy.numpy as np

    array_to_store = np.zeros(100)

    def update(_i_):
        array_to_store[_i_] = np.random.random()

    neu = bp.NeuType('test', steps=update, ...)

The update function in defined ``neu`` will not work actually. 
``array_to_store`` will not be modified.
Instead, you can update ``array_to_store`` by passing it into the function 
as the argument:

.. code-block:: python

    def update(_i_, array_to_store):
        array_to_store[_i_] = np.random.random()

    neu = bp.NeuType('test', steps=update, ...)

This will work.


Avoid containers
----------------

JIT compilers (like Numba, JAX) support containers, such as ``dict``, 
``namedtuple``. However, the computation based in containers will greatly 
reduces the performance. 


Continue .....


