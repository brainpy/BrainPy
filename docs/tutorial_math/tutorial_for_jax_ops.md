

```python

for i in range(self.pre.size[0]):
  if self.pre.spike[i] > 0:
    self.t_last_pre_spike[i] = _t
```




```python
import brainpy.math as bm
t_last_pre_spike : (num_pre, num_post)
self.pre.spike: (num_pre,)
_t : scalar

t_last_pre_spike = bm.reshape(bm.where(self.pre.spike, _t, 0.), (-1, 1))
```

