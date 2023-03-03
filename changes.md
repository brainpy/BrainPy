# Change from Version 2.3.4 to Version 2.3.5


This release (under the branch of ``brainpy=2.3.x``) continues to add supports for brain-inspired computation.


## New Features


### 1. ``brainpy.share`` for sharing data across submodules

In this release, we abstract the shared data as a ``brainpy.share`` object. 

This object together with ``brainpy.Delay`` we will introduce below 
constitute the support that enable to define SNN models like ANN ones.


### 2. ``brainpy.Delay`` for delay processing

``Delay`` is abstracted as a dynamical system, which can be updated / retrieved by users. 

```python
import brainpy as bp

class EINet(bp.DynamicalSystemNS):
  def __init__(self, scale=1.0, e_input=20., i_input=20., delay=None):
    super().__init__()

    self.bg_exc = e_input
    self.bg_inh = i_input

    # network size
    num_exc = int(3200 * scale)
    num_inh = int(800 * scale)

    # neurons
    pars = dict(V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                V_initializer=bp.init.Normal(-55., 2.), input_var=False)
    self.E = bp.neurons.LIF(num_exc, **pars)
    self.I = bp.neurons.LIF(num_inh, **pars)

    # synapses
    we = 0.6 / scale  # excitatory synaptic weight (voltage)
    wi = 6.7 / scale  # inhibitory synaptic weight
    self.E2E = bp.experimental.Exponential(
      bp.conn.FixedProb(0.02, pre=self.E.size, post=self.E.size),
      g_max=we, tau=5., out=bp.experimental.COBA(E=0.)
    )
    self.E2I = bp.experimental.Exponential(
      bp.conn.FixedProb(0.02, pre=self.E.size, post=self.I.size, ),
      g_max=we, tau=5., out=bp.experimental.COBA(E=0.)
    )
    self.I2E = bp.experimental.Exponential(
      bp.conn.FixedProb(0.02, pre=self.I.size, post=self.E.size),
      g_max=wi, tau=10., out=bp.experimental.COBA(E=-80.)
    )
    self.I2I = bp.experimental.Exponential(
      bp.conn.FixedProb(0.02, pre=self.I.size, post=self.I.size),
      g_max=wi, tau=10., out=bp.experimental.COBA(E=-80.)
    )
    self.delayE = bp.Delay(self.E.spike, entries={'E': delay})
    self.delayI = bp.Delay(self.I.spike, entries={'I': delay})

  def update(self):
    e_spike = self.delayE.at('E')
    i_spike = self.delayI.at('I')
    e_inp = self.E2E(e_spike, self.E.V) + self.I2E(i_spike, self.E.V) + self.bg_exc
    i_inp = self.I2I(i_spike, self.I.V) + self.E2I(e_spike, self.I.V) + self.bg_inh
    self.delayE(self.E(e_inp))
    self.delayI(self.I(i_inp))

```



### 3.  ``brainpy.checkpoints.save_pytree`` and ``brainpy.checkpoints.load_pytree`` for saving/loading target from the filename

Now we can directly use ``brainpy.checkpoints.save_pytree`` to save a 
network state into the filepath we specified. 

Similarly, we can use ``brainpy.checkpoints.load_pytree`` to load 
states from the given file path.


### 4. More ANN layers


- brainpy.layers.ConvTranspose1d
- brainpy.layers.ConvTranspose2d
- brainpy.layers.ConvTranspose3d
- brainpy.layers.Conv1dLSTMCell
- brainpy.layers.Conv2dLSTMCell
- brainpy.layers.Conv3dLSTMCell


### 5. More compatible dense operators

PyTorch operators:

- brainpy.math.Tensor
- brainpy.math.flatten
- brainpy.math.cat
- brainpy.math.abs
- brainpy.math.absolute
- brainpy.math.acos
- brainpy.math.arccos
- brainpy.math.acosh
- brainpy.math.arccosh
- brainpy.math.add
- brainpy.math.addcdiv
- brainpy.math.addcmul
- brainpy.math.angle
- brainpy.math.asin
- brainpy.math.arcsin
- brainpy.math.asinh
- brainpy.math.arcsin
- brainpy.math.atan
- brainpy.math.arctan
- brainpy.math.atan2
- brainpy.math.atanh


TensorFlow operators:

- brainpy.math.concat
- brainpy.math.reduce_sum
- brainpy.math.reduce_max
- brainpy.math.reduce_min
- brainpy.math.reduce_mean
- brainpy.math.reduce_all
- brainpy.math.reduce_any
- brainpy.math.reduce_logsumexp
- brainpy.math.reduce_prod
- brainpy.math.reduce_std
- brainpy.math.reduce_variance
- brainpy.math.reduce_euclidean_norm
- brainpy.math.unsorted_segment_sqrt_n
- brainpy.math.segment_mean
- brainpy.math.unsorted_segment_sum
- brainpy.math.unsorted_segment_prod
- brainpy.math.unsorted_segment_max
- brainpy.math.unsorted_segment_min
- brainpy.math.unsorted_segment_mean
- brainpy.math.segment_sum
- brainpy.math.segment_prod
- brainpy.math.segment_max
- brainpy.math.segment_min
- brainpy.math.clip_by_value
- brainpy.math.cast


### Others

- Remove the hard requirements of ``brainpylib`` and ``numba``.

