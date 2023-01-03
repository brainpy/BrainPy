# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
import numpy as np

import brainpy as bp
import brainpy.math as bm

conv_init = bp.init.KaimingNormal(mode='fan_out', scale=2 ** 0.5)
dense_init = bp.init.Normal(0, 0.01)


@jax.custom_vjp
def replace(spike, rate):
  return rate


def replace_fwd(spike, rate):
  return replace(spike, rate), ()


def replace_bwd(res, g):
  return g, g


replace.defvjp(replace_fwd, replace_bwd)


class ScaledWSConv2d(bp.layers.Conv2d):
  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               stride=1,
               padding=0,
               groups=1,
               b_initializer=bp.init.ZeroInit(),
               gain=True,
               eps=1e-4):
    super(ScaledWSConv2d, self).__init__(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         groups=groups,
                                         w_initializer=conv_init,
                                         b_initializer=b_initializer)
    bp.check.is_subclass(self.mode, bm.TrainingMode)
    if gain:
      self.gain = bm.TrainVar(jnp.ones(self.out_channels, 1, 1, 1))
    else:
      self.gain = None
    self.eps = eps

  def update(self, *args):
    assert self.mask is None
    x = args[0] if len(args) == 1 else args[1]
    self._check_input_dim(x)
    w = self.w.value
    fan_in = np.prod(w.shape[1:])
    mean = jnp.mean(w, axis=[1, 2, 3], keepdims=True)
    var = jnp.var(w, axis=[1, 2, 3], keepdims=True)
    w = (w - mean) / ((var * fan_in + self.eps) ** 0.5)
    if self.gain is not None:
      w = w * self.gain
    y = jax.lax.conv_general_dilated(lhs=bm.as_jax(x),
                                     rhs=bm.as_jax(w),
                                     window_strides=self.stride,
                                     padding=self.padding,
                                     lhs_dilation=self.lhs_dilation,
                                     rhs_dilation=self.rhs_dilation,
                                     feature_group_count=self.groups,
                                     dimension_numbers=self.dimension_numbers)
    return y if self.b is None else (y + self.b.value)


class ScaledWSLinear(bp.layers.Dense):
  def __init__(self,
               in_features,
               out_features,
               b_initializer=bp.init.ZeroInit(),
               gain=True,
               eps=1e-4):
    super(ScaledWSLinear, self).__init__(num_in=in_features,
                                         num_out=out_features,
                                         W_initializer=dense_init,
                                         b_initializer=b_initializer)
    bp.check.is_subclass(self.mode, bm.TrainingMode)
    if gain:
      self.gain = bm.TrainVar(jnp.ones(1, self.num_out))
    else:
      self.gain = None
    self.eps = eps

  def update(self, s, x):
    fan_in = self.W.shape[0]
    mean = jnp.mean(self.W.value, axis=0, keepdims=True)
    var = jnp.var(self.W.value, axis=0, keepdims=True)
    weight = (self.W.value - mean) / ((var * fan_in + self.eps) ** 0.5)
    if self.gain is not None:
      weight = weight * self.gain
    if self.b is not None:
      return x @ weight + self.b
    else:
      return x @ weight


class Scale(bp.layers.Layer):
  def __init__(self, scale: float):
    super(Scale, self).__init__()
    self.scale = scale

  def update(self, s, x):
    return x * self.scale


class WrappedSNNOp(bp.layers.Layer):
  def __init__(self, op):
    super(WrappedSNNOp, self).__init__()
    self.op = op

  def update(self, s, x):
    if s['require_wrap']:
      spike, rate = jnp.split(x, 2, axis=0)
      out = jax.lax.stop_gradient(self.op(spike))
      in_for_grad = replace(spike, rate)
      out_for_grad = self.op(in_for_grad)
      output = replace(out_for_grad, out)
      return output
    else:
      return self.op(x)


class OnlineSpikingVGG(bp.DynamicalSystem):
  def __init__(
      self,
      cfg,
      weight_standardization=True,
      num_classes=1000,
      neuron_model: callable = None,
      neuron_pars: dict = None,
      light_classifier=True,
      batch_norm=False,
      grad_with_rate: bool = False,
      fc_hw: int = 3,
      c_in: int = 3
  ):
    super(OnlineSpikingVGG, self).__init__()

    if neuron_pars is None: neuron_pars = dict()
    self.neuron_pars = neuron_pars
    self.neuron_model = neuron_model
    self.grad_with_rate = grad_with_rate
    self.fc_hw = fc_hw

    self.features = self.make_layers(cfg=cfg,
                                     in_channels=c_in,
                                     weight_standardization=weight_standardization,
                                     batch_norm=batch_norm)
    if light_classifier:
      self.avgpool = bp.layers.AdaptiveAvgPool2d((self.fc_hw, self.fc_hw))
      if self.grad_with_rate:
        self.classifier = WrappedSNNOp(bp.layers.Dense(512 * (self.fc_hw ** 2),
                                                       num_classes,
                                                       W_initializer=dense_init))
      else:
        self.classifier = bp.layers.Dense(512 * (self.fc_hw ** 2),
                                          num_classes,
                                          W_initializer=dense_init)
    else:
      self.avgpool = bp.layers.AdaptiveAvgPool2d((7, 7))
      if self.grad_with_rate:
        self.classifier = bp.Sequential(
          WrappedSNNOp(ScaledWSLinear(512 * 7 * 7, 4096)),
          neuron_model((4096,), **self.neuron_pars, neuron_dropout=0.0),
          Scale(2.74),
          bp.layers.Dropout(0.5),
          WrappedSNNOp(ScaledWSLinear(4096, 4096)),
          neuron_model((4096,), **self.neuron_pars, neuron_dropout=0.0),
          Scale(2.74),
          bp.layers.Dropout(0.5),
          WrappedSNNOp(bp.layers.Dense(4096, num_classes, W_initializer=dense_init)),
        )
      else:
        self.classifier = bp.Sequential(
          ScaledWSLinear(512 * 7 * 7, 4096),
          neuron_model((4096,), **self.neuron_pars, neuron_dropout=0.0),
          Scale(2.74),
          bp.layers.Dropout(0.5),
          ScaledWSLinear(4096, 4096),
          neuron_model((4096,), **self.neuron_pars, neuron_dropout=0.0),
          Scale(2.74),
          bp.layers.Dropout(0.5),
          bp.layers.Dense(4096, num_classes, W_initializer=dense_init),
        )

  def update(self, s, x):
    if self.grad_with_rate and s['fit']:
      s['require_wrap'] = True
      s['output_type'] = 'spike_rate'
      x = self.features(s, x)
      x = self.avgpool(s, x)
      x = bm.flatten(x, 1)
      x = self.classifier(s, x)
    else:
      s['require_wrap'] = False
      x = self.features(s, x)
      x = self.avgpool(s, x)
      x = bm.flatten(x, 1)
      x = self.classifier(s, x)
    return x

  def make_layers(
      self,
      cfg,
      in_channels,
      weight_standardization: bool = True,
      batch_norm: bool = False,
  ):
    layers = []
    first_conv = True
    use_stride_2 = False
    for v in cfg:
      if v == 'M':
        layers.append(bp.layers.AvgPool(kernel_size=2, stride=2))
      elif v == 'S':
        use_stride_2 = True
      else:
        if use_stride_2:
          stride = 2
          use_stride_2 = False
        else:
          stride = 1
        if weight_standardization:
          if first_conv:
            conv2d = ScaledWSConv2d(in_channels, v,
                                    kernel_size=3,
                                    padding=1,
                                    stride=stride)
            first_conv = False
          else:
            if self.grad_with_rate:
              conv2d = WrappedSNNOp(ScaledWSConv2d(in_channels, v,
                                                   kernel_size=3,
                                                   padding=1,
                                                   stride=stride))
            else:
              conv2d = ScaledWSConv2d(in_channels, v,
                                      kernel_size=3,
                                      padding=1,
                                      stride=stride)
          layers += [conv2d, self.neuron_model(**self.neuron_pars), Scale(2.74)]
        else:
          if first_conv:
            conv2d = bp.layers.Conv2d(in_channels, v,
                                      kernel_size=3,
                                      padding=1,
                                      stride=stride,
                                      w_initializer=conv_init, )
            first_conv = False
          else:
            if self.grad_with_rate:
              conv2d = WrappedSNNOp(bp.layers.Conv2d(in_channels, v,
                                                     kernel_size=3,
                                                     padding=1,
                                                     stride=stride,
                                                     w_initializer=conv_init, ))
            else:
              conv2d = bp.layers.Conv2d(in_channels, v,
                                        kernel_size=3,
                                        padding=1,
                                        stride=stride,
                                        w_initializer=conv_init, )
          if batch_norm:
            bn = bp.layers.BatchNorm2d(v, momentum=0.9)
            layers += [conv2d, bn, self.neuron_model(**self.neuron_pars)]
          else:
            layers += [conv2d, self.neuron_model(**self.neuron_pars), Scale(2.74)]
        in_channels = v
    return bp.Sequential(*layers)


class OnlineSpikingVGGF(bp.DynamicalSystem):
  def __init__(self,
               cfg,
               weight_standardization=True,
               num_classes=1000,
               neuron_model: callable = None,
               light_classifier=True,
               batch_norm=False,
               grad_with_rate=False,
               fc_hw: int = 3,
               c_in: int = 3,
               **kwargs):
    super(OnlineSpikingVGGF, self).__init__()
    self.neuron_model = neuron_model
    self.grad_with_rate = grad_with_rate
    self.fc_hw = fc_hw
    self.conv1, self.features = self.make_layers(
      cfg=cfg,
      in_channels=c_in,
      grad_with_rate=grad_with_rate,
      weight_standardization=weight_standardization,
      batch_norm=batch_norm,
    )

    # feedback connections
    scale_factor = 1
    for v in cfg:
      if v == 'M' or v == 'S':
        scale_factor *= 2
    self.up = nn.Upsample(scale_factor=scale_factor, mode='nearest')
    self.fb_conv = bp.layers.Conv2d(cfg[-1], cfg[0], kernel_size=3, padding=1, stride=1,
                                    w_initializer=bp.init.ZeroInit())
    if self.grad_with_rate:
      self.fb_conv = WrappedSNNOp(self.fb_conv)

    if light_classifier:
      self.avgpool = bp.layers.AdaptiveAvgPool2d((self.fc_hw, self.fc_hw))
      if self.grad_with_rate:
        self.classifier = WrappedSNNOp(bp.layers.Dense(512 * (self.fc_hw ** 2), num_classes, W_initializer=dense_init))
      else:
        self.classifier = bp.layers.Dense(512 * (self.fc_hw ** 2), num_classes, W_initializer=dense_init)
    else:
      self.avgpool = bp.layers.AdaptiveAvgPool2d((7, 7))
      if self.grad_with_rate:
        self.classifier = bp.Sequential(
          WrappedSNNOp(ScaledWSLinear(512 * 7 * 7, 4096)),
          neuron_model(**kwargs, neuron_dropout=0.0),
          Scale(2.74),
          bp.layers.Dropout(0.5),
          WrappedSNNOp(ScaledWSLinear(4096, 4096)),
          neuron_model(**kwargs, neuron_dropout=0.0),
          Scale(2.74),
          bp.layers.Dropout(0.5),
          WrappedSNNOp(bp.layers.Dense(4096, num_classes, W_initializer=dense_init)),
        )
      else:
        self.classifier = bp.Sequential(
          ScaledWSLinear(512 * 7 * 7, 4096),
          neuron_model(**kwargs, neuron_dropout=0.0),
          Scale(2.74),
          bp.layers.Dropout(0.5),
          ScaledWSLinear(4096, 4096),
          neuron_model(**kwargs, neuron_dropout=0.0),
          Scale(2.74),
          bp.layers.Dropout(0.5),
          bp.layers.Dense(4096, num_classes, W_initializer=dense_init),
        )

    self.fb_features = bp.init.variable_(bm.zeros, xxx, 1)

  def reset_state(self, batch_size=1):
    self.fb_features.value = bp.init.variable_(bm.zeros, xxx, batch_axis)
    for node in self.nodes().subset(bp.DynamicalSystem).unique():
      node.reset_state(batch_size)

  def update(self, s, x):
    if self.grad_with_rate and s['fit']:
      s['output_type'] = 'spike_rate'
      s['require_wrap'] = True
      x = self.conv1(s, x) + self.fb_conv(s, self.up(s, self.fb_features))
      x = self.features(s, x)
      self.fb_features = jax.lax.stop_gradient(x)
      x = self.avgpool(s, x)
      x = bm.flatten(x, 1)
      x = self.classifier(s, x)
    else:
      s['require_wrap'] = False
      if self.grad_with_rate:
        x = self.conv1(s, x) + self.fb_conv(s, self.up(s, self.fb_features))
      else:
        x = self.conv1(s, x) + self.fb_conv(s, self.up(s, self.fb_features))
      x = self.features(s, x)
      self.fb_features = jax.lax.stop_gradient(x)
      x = self.avgpool(s, x)
      x = bm.flatten(x, 1)
      x = self.classifier(s, x)
    return x

  def make_layers(self,
                  cfg,
                  in_channels,
                  grad_with_rate=False,
                  weight_standardization=True,
                  batch_norm=False,
                  **kwargs):
    layers = []
    first_conv = True
    use_stride_2 = False
    for v in cfg:
      if v == 'M':
        layers += [bp.layers.AvgPool2d(kernel_size=2, stride=2)]
      elif v == 'S':
        use_stride_2 = True
      else:
        if use_stride_2:
          stride = 2
          use_stride_2 = False
        else:
          stride = 1
        if weight_standardization:
          if first_conv:
            conv1 = ScaledWSConv2d(in_channels, v, kernel_size=3, padding=1, stride=stride)
            first_conv = False
            layers += [self.neuron_model(**kwargs), Scale(2.74)]
          else:
            if grad_with_rate:
              conv2d = WrappedSNNOp(ScaledWSConv2d(in_channels, v, kernel_size=3, padding=1, stride=stride))
            else:
              conv2d = ScaledWSConv2d(in_channels, v, kernel_size=3, padding=1, stride=stride)
            layers += [conv2d, self.neuron_model(**kwargs), Scale(2.74)]
        else:
          if first_conv:
            conv1 = bp.layers.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=stride)
            first_conv = False
            if batch_norm:
              bn = bp.layers.BatchNorm2d(v, momentum=0.9)
              layers += [bn, self.neuron_model(**kwargs)]
            else:
              layers += [self.neuron_model(**kwargs), Scale(2.74)]
          else:
            if grad_with_rate:
              conv2d = WrappedSNNOp(bp.layers.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=stride))
            else:
              conv2d = bp.layers.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=stride)
            if batch_norm:
              bn = bp.layers.BatchNorm2d(v, momentum=0.9)
              layers += [conv2d, bn, self.neuron_model(**kwargs)]
            else:
              layers += [conv2d, self.neuron_model(**kwargs), Scale(2.74)]
        in_channels = v
    return conv1, bp.Sequential(*layers)


class OnlineIFNode(bp.DynamicalSystem):
  def __init__(
      self,
      size,
      v_threshold: float = 1.,
      v_reset: float = None,
      f_surrogate=bm.surrogate.sigmoid,
      detach_reset: bool = True,
      track_rate: bool = True,
      neuron_dropout: float = 0.0,
      name: str = None,
      mode: bm.Mode = None
  ):
    super().__init__(name=name, mode=mode)
    bp.check.is_subclass(self.mode, bm.TrainingMode)

    self.size = bp.check.is_sequence(size, elem_type=int)
    self.f_surrogate = bp.check.is_callable(f_surrogate)
    self.detach_reset = detach_reset
    self.v_reset = v_reset
    self.v_threshold = v_threshold
    self.track_rate = track_rate
    self.dropout = neuron_dropout

    if self.dropout > 0.0:
      self.rng = bm.random.default_rng()
    self.reset_state(1)

  def reset_state(self, batch_size=1):
    self.v = bp.init.variable_(bm.zeros, self.size, batch_size)
    self.spike = bp.init.variable_(bm.zeros, self.size, batch_size)
    if self.track_rate:
      self.rate_tracking = bp.init.variable_(bm.zeros, self.size, batch_size)

  def update(self, s, x):
    # neuron charge
    self.v.value = jax.lax.stop_gradient(self.v.value) + x
    # neuron fire
    spike = self.f_surrogate(self.v.value - self.v_threshold)
    # spike reset
    spike_d = jax.lax.stop_gradient(spike) if self.detach_reset else spike
    if self.v_reset is None:
      self.v -= spike_d * self.v_threshold
    else:
      self.v.value = (1. - spike_d) * self.v + spike_d * self.v_reset
    # dropout
    if self.dropout > 0.0 and s['fit']:
      mask = self.rng.bernoulli(1 - self.dropout, self.v.shape) / (1 - self.dropout)
      spike = mask * spike
    self.spike.value = spike
    # spike track
    if self.track_rate:
      self.rate_tracking += jax.lax.stop_gradient(spike)
    # output
    if s['output_type'] == 'spike_rate':
      assert self.track_rate
      return jnp.concatenate([spike, self.rate_tracking.value])
    else:
      return spike


class OnlineLIFNode(bp.DynamicalSystem):
  def __init__(
      self,
      size,
      tau: float = 2.,
      decay_input: bool = False,
      v_threshold: float = 1.,
      v_reset: float = None,
      f_surrogate=bm.surrogate.sigmoid,
      detach_reset: bool = True,
      track_rate: bool = True,
      neuron_dropout: float = 0.0,
      name: str = None,
      mode: bm.Mode = None
  ):
    super().__init__(name=name, mode=mode)
    bp.check.is_subclass(self.mode, bm.TrainingMode)

    self.size = bp.check.is_sequence(size, elem_type=int)
    self.tau = tau
    self.decay_input = decay_input
    self.v_threshold = v_threshold
    self.v_reset = v_reset
    self.f_surrogate = f_surrogate
    self.detach_reset = detach_reset
    self.track_rate = track_rate
    self.dropout = neuron_dropout

    if self.dropout > 0.0:
      self.rng = bm.random.default_rng()

  def reset_state(self, batch_size=1):
    self.v = bp.init.variable_(bm.zeros, self.size, batch_size)
    self.spike = bp.init.variable_(bm.zeros, self.size, batch_size)
    if self.track_rate:
      self.rate_tracking = bp.init.variable_(bm.zeros, self.size, batch_size)

  def forward(self, s, x):
    # neuron charge
    if self.decay_input:
      x = x / self.tau
    if self.v_reset is None or self.v_reset == 0:
      self.v = jax.lax.stop_gradient(self.v.value) * (1 - 1. / self.tau) + x
    else:
      self.v = jax.lax.stop_gradient(self.v.value) * (1 - 1. / self.tau) + self.v_reset / self.tau + x
    # neuron fire
    spike = self.f_surrogate(self.v - self.v_threshold)
    # neuron reset
    spike_d = jax.lax.stop_gradient(spike) if self.detach_reset else spike
    if self.v_reset is None:
      self.v = self.v - spike_d * self.v_threshold
    else:
      self.v = (1. - spike_d) * self.v + spike_d * self.v_reset
    # dropout
    if self.dropout > 0.0 and s['fit']:
      mask = self.rng.bernoulli(1 - self.dropout, self.v.shape) / (1 - self.dropout)
      spike = mask * spike
    self.spike.value = spike
    # spike
    if self.track_rate:
      self.rate_tracking.value = jax.lax.stop_gradient(self.rate_tracking * (1 - 1. / self.tau) + spike)
    if s['output_type'] == 'spike_rate':
      assert self.track_rate
      return jnp.concatenate((spike, self.rate_tracking))
    else:
      return spike



