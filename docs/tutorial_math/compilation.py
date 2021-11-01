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
# # Compilation

# %% [markdown]
# In this section, we are going to talk about the concept of the code compilation to accelerate your model running performance. 

# %%
import brainpy as bp
import brainpy.math.jax as bm

bp.math.use_backend('jax')


# %% [markdown]
# ## ``jit()``

# %% [markdown]
# We have talked about the mechanism of [JIT compilation for class objects in NumPy backend](../quickstart/jit_compilation.ipynb#Mechanism-of-JIT-in-NumPy-backend). In this section, we try to understand how to apply JIT when you are using JAX backend. 

# %% [markdown]
# ``jax.jit()`` is excellent, while only supports [pure functions](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions). ``brainpy.math.jax.jit()`` is based on ``jax.jit()``, but it extends its ability to just-in-time compile your class objects. 

# %% [markdown]
# ### JIT for pure functions

# %% [markdown]
# First, ``brainpy.math.jax.jit()`` can just-in-time compile your functions. 

# %%
def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * bm.where(x > 0, x, alpha * bm.exp(x) - alpha)

x = bm.random.normal(size=(1000000,))

# %%
# %timeit selu(x)

# %%
selu_jit = bm.jit(selu) # jit accleration

# %timeit selu_jit(x)

# %% [markdown]
# ### JIT for class objects

# %% [markdown]
# Moreover, ``brainpy.math.jax.jit()`` is powerful to just-in-time compile your class objects. The [constraints](../quickstart/jit_compilation.ipynb) for class object JIT are:
#
# - The JIT target must be a subclass of ``brainpy.Base``.
# - Dynamically changed variables must be labeled as ``brainpy.math.Variable``.
# - Variable changes must be made in-place. 

# %%
class LogisticRegression(bp.Base):
    def __init__(self, dimension):
        super(LogisticRegression, self).__init__()

        # parameters
        self.dimension = dimension

        # variables
        self.w = bm.Variable(2.0 * bm.ones(dimension) - 1.3)

    def __call__(self, X, Y):
        u = bm.dot(((1.0 / (1.0 + bm.exp(-Y * bm.dot(X, self.w))) - 1.0) * Y), X)
        self.w[:] = self.w - u

num_dim, num_points = 10, 200000
points = bm.random.random((num_points, num_dim))
labels = bm.random.random(num_points)

# %%
lr = LogisticRegression(10)

# %%
# %timeit lr(points, labels)

# %%
lr_jit = bm.jit(lr)

# %timeit lr_jit(points, labels)

# %% [markdown]
# ### JIT mechanism

# %% [markdown]
# The mechanism of JIT compilation is that BrainPy automatically transforms your class methods into functions. 
#
# ``brainpy.math.jax.jit()`` receives a ``dyn_vars`` argument, which denotes the dynamically changed variables. If you do not provide it, BrainPy will automatically detect them by calling ``Base.vars()``. Once get "dyn_vars", BrainPy will treat "dyn_vars" as function arguments, thus making them able to dynamically change. 

# %%
import types

isinstance(lr_jit, types.FunctionType)  # "lr" is class, while "lr_jit" is a function


# %% [markdown]
# Therefore, the secrete of ``brainpy.math.jax.jit()`` is providing "dyn_vars". No matter your target is a class object, a method in the class object, or a pure function, if there are dynamically changed variables, you just pack them into ``brainpy.math.jax.jit()`` as "dyn_vars". Then, all the compilation and acceleration will be handled by BrainPy automatically. 

# %% [markdown]
# ### Example 1: JIT a class method

# %%
class Linear(bp.Base):
    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()
        self.w = bm.TrainVar(bm.random.random((n_in, n_out)))
        self.b = bm.TrainVar(bm.zeros(n_out))
    
    def update(self, x):
        return x @ self.w + self.b


# %%
x = bm.zeros(10)
l = Linear(10, 3)

# %% [markdown]
# This time, we mark "w" and "b" as dynamically changed variables. 

# %%
update1 = bm.jit(l.update, dyn_vars=[l.w, l.b])  # make 'w' and 'b' dynamically change
update1(x)  # x is 0., b is 0., therefore y is 0.

# %%
l.b[:] = 1.  # change b to 1, we expect y will be 1 too

update1(x)

# %% [markdown]
# This time, we only mark "w" as dynamically changed variables. We will find also modify "b", the results will not change. 

# %%
update2 = bm.jit(l.update, dyn_vars=[l.w])  # make 'w' dynamically change

update2(x)

# %%
l.b[:] = 2.  # change b to 2, while y will not be 2
update2(x)

# %% [markdown]
# ### Example 2: JIT a function

# %% [markdown]
# Now, we change the above "Linear" object to a function. 

# %%
n_in = 10;  n_out = 3

w = bm.TrainVar(bm.random.random((n_in, n_out)))
b = bm.TrainVar(bm.zeros(n_out))

def update(x):
    return x @ w + b


# %% [markdown]
# If we do not provide ``dyn_vars``, "w" and "b" will be compiled as constant values. 

# %%
update1 = bm.jit(update)
update1(x)

# %%
b[:] = 1.  # modify the value of 'b' will not 
           # change the result, because in the 
           # jitted function, 'b' is already 
           # a constant
update1(x)

# %% [markdown]
# Provide "w" and "b" as ``dyn_vars`` will make them dynamically changed again. 

# %%
update2 = bm.jit(update, dyn_vars=(w, b))
update2(x)

# %%
b[:] = 2.  # change b to 2, while y will not be 2
update2(x)


# %% [markdown]
# ### RandomState

# %% [markdown]
# We have talked about RandomState in [Variables](./variables.ipynb) section. We said that it is also a Variable. Therefore, if your functions have used the default RandomState (``brainpy.math.jax.random.DEFAULT``), you should add it into the ``dyn_vars`` scope of the function. Otherwise, they will be treated as constants and the jitted function will always return the same value. 

# %%
def function():
    return bm.random.normal(0, 1, size=(10,))


# %%
f1 = bm.jit(function)

f1() == f1()

# %% [markdown]
# The correct way to make JIT for this function is:

# %%
bm.random.seed(1234)

f2 = bm.jit(function, dyn_vars=bm.random.DEFAULT)

f2() == f2()


# %% [markdown]
# ### Example 3: JIT a neural network

# %% [markdown]
# Now, let's use SGD to train a neural network with JIT acceleration. Here we will use the autograd function ``brainpy.math.jax.grad()``, which will be detailed out in [the next section](./differentiation.ipynb).

# %%
class LinearNet(bp.Base):
    def __init__(self, n_in, n_out):
        super(LinearNet, self).__init__()

        # weights
        self.w = bm.TrainVar(bm.random.random((n_in, n_out)))
        self.b = bm.TrainVar(bm.zeros(n_out))
        self.r = bm.TrainVar(bm.random.random((n_out, 1)))
    
    def update(self, x):
        h = x @ self.w + self.b
        return h @ self.r
    
    def loss(self, x, y):
        predict = self.update(x)
        return bm.mean((predict - y) ** 2)


ln = LinearNet(100, 200)

# provide the variables want to update
opt = bm.optimizers.SGD(lr=1e-6, train_vars=ln.vars()) 

# provide the variables require graidents
f_grad = bm.grad(ln.loss, grad_vars=ln.vars(), return_value=True)  


def train(X, Y):
    grads, loss = f_grad(X, Y)
    opt.update(grads)
    return loss

# JIT the train function 
train_jit = bm.jit(train, dyn_vars=ln.vars() + opt.vars())


# %%
xs = bm.random.random((1000, 100))
ys = bm.random.random((1000, 1))

for i in range(30):
    loss  = train_jit(xs, ys)
    print(f'Train {i}, loss = {loss:.2f}')


# %% [markdown]
# ### Static arguments

# %% [markdown]
# Static arguments are arguments that are treated as static/constant in the jitted function. 
#
# Numerical arguments used in condition syntax (bool value or resulting bool value), and strings must be marked as static. Otherwise, an error will raise. 

# %%
@bm.jit
def f(x):
  if x < 3:  # this will cause error
    return 3. * x ** 2
  else:
    return -4 * x


# %%
try:
    f(1.)
except Exception as e:
    print(type(e), e)


# %% [markdown]
# Simply speaking, arguments resulting boolean values must be declared as static arguments. In ``brainpy.math.jax.jit()`` function, if can set the names of static arguments. 

# %%
def f(x):
  if x < 3:  # this will cause error
    return 3. * x ** 2
  else:
    return -4 * x

f_jit = bm.jit(f, static_argnames=('x', ))

# %%
f_jit(x=1.)

# %% [markdown]
# However, it's worthy noting that calling the jitted function with different values for these static arguments will trigger recompilation. Therefore, declaring static arguments may be suitable to the following situations:
#
# 1. Boolean arguments.
# 2. Arguments only have several possible values. 
#
# If the argument value change significantly, you'd better not to declare it as static. 
#
# For more information, please refer to [jax.jit](https://jax.readthedocs.io/en/latest/jax.html?highlight=jit#jax.jit) API.

# %% [markdown]
# ## ``vmap()``

# %% [markdown]
# Coming soon. 

# %% [markdown]
# ## ``pmap()``

# %% [markdown]
# Coming soon. 
