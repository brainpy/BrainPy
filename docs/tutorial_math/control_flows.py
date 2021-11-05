# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: brainpy
#     language: python
#     name: brainpy
# ---

# %% [markdown]
# # Control Flows

# %% [markdown]
# In this section, we are going to talk about how to build structured control flows in 'jax' backend. These control flows include 
#
# - *for loop* syntax, 
# - *while loop* syntax,  
# - and *condition* syntax. 

# %%
import brainpy as bp
import brainpy.math.jax as bm

bp.math.use_backend('jax')


# %% [markdown]
# In JAX, the control flow syntaxes are not easy to use. Users must transform the intuitive Python control flows into [structured control flows](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#structured-control-flow-primitives). 
#
# Based on ``JaxArray`` provided in BrainPy, we try to present a better syntax to make control flows. 

# %% [markdown]
# ## ``make_loop()``

# %% [markdown]
# ``brainpy.math.jax.make_loop()`` is used to generate a for-loop function when you are using ``JaxArray``. 
#
# Let's image your requirement: you are using several JaxArray (grouped as ``dyn_vars``) to implement your body function "body\_fun", and you want to gather the history values of several of them (grouped as ``out_vars``). Sometimes, your body function return something, and you also want to gather the return values. With Python syntax, your requirement is equivalent to 
#
# ```python
#
# def for_loop_function(body_fun, dyn_vars, out_vars, xs):
#   ys = []
#   for x in xs:
#     # 'dyn_vars' and 'out_vars' 
#     # are updated in 'body_fun()'
#     results = body_fun(x)
#     ys.append([out_vars, results])
#   return ys
#
# ```

# %% [markdown]
# In BrainPy, using ``brainpy.math.jax.make_loop()`` you can define this logic like:
#
# ```python
#
# loop_fun = brainpy.math.jax.make_loop(body_fun, dyn_vars, out_vars, has_return=False)
#
# hist_values = loop_fun(xs)
# ```
#
# Or, 
#
# ```python
#
# loop_fun = brainpy.math.jax.make_loop(body_fun, dyn_vars, out_vars, has_return=True)
#
# hist_of_vars, hist_of_return_vars = loop_fun(xs)
# ```
#

# %% [markdown]
# Let's implement a recurrent network to illustrate how to use this function. 

# %%
class RNN(bp.DynamicalSystem):
  def __init__(self, n_in, n_h, n_out, n_batch, g=1.0, **kwargs):
    super(RNN, self).__init__(**kwargs)

    # parameters
    self.n_in = n_in
    self.n_h = n_h
    self.n_out = n_out
    self.n_batch = n_batch
    self.g = g

    # weights
    self.w_ir = bm.TrainVar(bm.random.normal(scale=1 / n_in ** 0.5, size=(n_in, n_h)))
    self.w_rr = bm.TrainVar(bm.random.normal(scale=g / n_h ** 0.5, size=(n_h, n_h)))
    self.b_rr = bm.TrainVar(bm.zeros((n_h,)))
    self.w_ro = bm.TrainVar(bm.random.normal(scale=1 / n_h ** 0.5, size=(n_h, n_out)))
    self.b_ro = bm.TrainVar(bm.zeros((n_out,)))

    # variables
    self.h = bm.Variable(bm.random.random((n_batch, n_h)))

    # function
    self.predict = bm.make_loop(self.cell,
                                dyn_vars=self.vars(),
                                out_vars=self.h,
                                has_return=True)

  def cell(self, x):
    self.h[:] = bm.tanh(self.h @ self.w_rr + x @ self.w_ir + self.b_rr)
    o = self.h @ self.w_ro + self.b_ro
    return o


rnn = RNN(n_in=10, n_h=100, n_out=3, n_batch=5)

# %% [markdown]
# In the above `RNN` model, we define a body function ``RNN.cell`` for later for-loop over input values. The loop function is defined as ``self.predict`` with ``bm.make_loop()``. We care about the history values of "self.h" and the readout value "o", so we set ``out_vars = self.h`` and ``has_return=True``.  

# %%
xs = bm.random.random((100, rnn.n_in))
hist_h, hist_o = rnn.predict(xs)


# %%
hist_h.shape  # the shape should be (num_time,) + h.shape

# %%
hist_o.shape  # the shape should be (num_time, ) + o.shape

# %% [markdown]
# If you have multiple input values, you should wrap them as a container and call the loop function with ``loop_fun(xs)``, where "xs" can be a JaxArray, list/tuple/dict of JaxArray. For examples: 

# %%
a = bm.zeros(10)

def body(x):
    x1, x2 = x  # "x" is a tuple/list of JaxArray
    a.value += (x1 + x2)

loop = bm.make_loop(body, dyn_vars=[a], out_vars=a)
loop(xs=[bm.arange(10), bm.ones(10)])

# %%
a = bm.zeros(10)

def body(x):  # "x" is a dict of JaxArray
    a.value += x['a'] + x['b']

loop = bm.make_loop(body, dyn_vars=[a], out_vars=a)
loop(xs={'a': bm.arange(10), 'b': bm.ones(10)})

# %% [markdown]
# ``dyn_vars``, ``out_vars``, ``xs`` and body function returns can be arrays with the container structure like tuple/list/dict. The history output values will preserve the container structure of ``out_vars``and body function returns. If ``has_return=True``, the loop function will return a tuple of ``(hist_of_out_vars, hist_of_fun_returns)``. If no values are interested, please set ``out_vars=None``, and the loop function only returns ``hist_of_out_vars``. 

# %% [markdown]
# ## ``make_while()``

# %% [markdown]
# ``brainpy.math.jax.make_while()`` is used to generate a while-loop function when you are using ``JaxArray``. It supports the following loop logic:
#
# ```python
#
# while condition:
#     statements
# ```
#
# When using ``brainpy.math.jax.make_while()`` , *condition* should be wrapped as a ``cond_fun`` function which returns a boolean value, and *statements* should be packed as a ``body_fun`` function which does not support return values: 
#
# ```python
#
# while cond_fun(x):
#     body_fun(x)
# ```
#
# where ``x`` is the external input which is not iterated. All the iterated variables should be marked as ``JaxArray``. All ``JaxArray`` used in ``cond_fun`` and ``body_fun`` should be declared in a ``dyn_vars`` variable. 

# %% [markdown]
# Let's look an example:

# %%
i = bm.zeros(1)
counter = bm.zeros(1)

def cond_f(x): 
    return i[0] < 10

def body_f(x):
    i.value += 1.
    counter.value += i

loop = bm.make_while(cond_f, body_f, dyn_vars=[i, counter])

# %% [markdown]
# In the above, we try to implement a sum from 0 to 10. We use two JaxArray ``i`` and ``counter``. 

# %%
loop()

# %%
counter

# %%
i

# %% [markdown]
# ## ``make_cond()``

# %% [markdown]
# ``brainpy.math.jax.make_cond()`` is used to generate a condition function when you are using ``JaxArray``. It supports the following condition logic:
#
# ```python
#
# if True:
#     true statements 
# else: 
#     false statements
# ```
#
# When using ``brainpy.math.jax.make_cond()`` , *true statements* should be wrapped as a ``true_fun`` function which implements logics under true assert (no return), and *false statements* should be wrapped as a ``false_fun`` function which implements logics under false assert (also does not support return values): 
#
# ```python
#
# if True:
#     true_fun(x)
# else:
#     false_fun(x)
# ```
#
# All the ``JaxArray`` used in ``true_fun`` and ``false_fun`` should be declared in the ``dyn_vars`` argument. ``x`` is also used to receive the external input value. 

# %% [markdown]
# Let's make a try:

# %%
a = bm.zeros(2)
b = bm.ones(2)

def true_f(x):  a.value += 1

def false_f(x): b.value -= 1

cond = bm.make_cond(true_f, false_f, dyn_vars=[a, b])

# %% [markdown]
# Here, we have two tensors. If true, tensor ``a`` add 1; if false, tensor ``b`` subtract 1. 

# %%
cond(pred=True)

a, b

# %%
cond(True)

a, b

# %%
cond(False)

a, b

# %%
cond(False)

a, b

# %% [markdown]
# Or, we define a conditional case which depends on the external input. 

# %%
a = bm.zeros(2)
b = bm.ones(2)

def true_f(x):  a.value += x

def false_f(x): b.value -= x

cond = bm.make_cond(true_f, false_f, dyn_vars=[a, b])

# %%
cond(True, 10.)

a, b

# %%
cond(False, 5.)

a, b
