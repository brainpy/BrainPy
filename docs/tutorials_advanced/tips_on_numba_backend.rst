Tips on Numba Backend
=====================

Minimal knowledge about JIT programming will make you code much
faster codes on BrainPy with JIT backend, such as Numba.


Global variable cannot be updated
---------------------------------

JIT compilers treat global variables as compile-time constants, which
means during the updating of step functions, global variables will not 
be updated. Anything you want to modify must be a `self.` data, or must
passed as an argument in the step functions. For example, if you want to
store some value in a 1D array at every calling of the step function:

.. code-block:: python

    import brainpy as bp
    import numpy as np

    array_to_store = np.zeros(100)

    class Example(bp.NeuGroup):

        def update(self, _i):
            array_to_store[_i] = np.random.random()

The update function in defined ``Example`` class will not work actually.
``array_to_store`` will not be modified, although it is an array.
This is because, in the JIT compilation, such global variables will
be treated as the static variables. Instead, you can update
``array_to_store`` by making it as an self data :

.. code-block:: python

    class Example(bp.NeuGroup):
        def __init__(self, ):
            self.array_to_store = np.zeros(100)

        def update(self, _i):
            array_to_store[_i] = np.random.random()

This will work.


Avoid containers
----------------

JIT compilers (like Numba, JAX) support containers, such as ``dict``, 
``namedtuple``. However, the computation based on containers will greatly
reduces the performance. Therefore, avoid use containers to store the
variables. Instead, you should always transform the data storage in the
``dict`` and ``namedtuple`` into the arrays. For example, compared to store
the spikes in a ``list`` structure,

.. code-block:: python

    class Example(bp.NeuGroup):
        def __init__(self, ):
            self.V = np.zeros(10)
            self.spike_hists = []

        def update(self):
            spike_ids = np.where(self.V > 10.)
            self.spike_hists.extend(spike_ids)


you should better use the array:


.. code-block:: python

    class Example(bp.NeuGroup):
        def __init__(self, ):
            self.V = np.zeros(10)
            self.spike_hists = np.zeros((1000, 10))

        def update(self, _i):
            self.spike_hists[_i] = self.V > 10.


Avoid changing the variable type
--------------------------------

JIT compiler relies on type inference to boost performance. Therefore,
keep the variable type consistent. An analogous "type-stability" problem
is like this:

.. code-block:: python

    class Example(bp.NeuGroup):

        def update(self, _i):
            x = 1
            for i in range(10):
                x /= 3.
            ...

Local variable `x` starts as an integer, and after one loop iteration becomes
a floating-point number (the result of / operator). This makes it more difficult
for the compiler to optimize the body of the loop.


Other tips please see `Performance Tips <https://numba.pydata.org/numba-doc/latest/user/performance-tips.html>`_ of Numba.
