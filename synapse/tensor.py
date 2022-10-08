import numpy as np
import inspect

class Tensor:
  def __init__(self, data):
    self.data = np.array(data, dtype=np.float32)
    self.grad = None
    self.ctx = None

  @property
  def shape(self):
    return self.data.shape

  @property
  def T(self):
    return self.data.T

  def backward(self):
    if self.ctx is None:
      return

    if self.grad is None:
      self.grad = np.ones(self.shape)

    grads = self.ctx.backward(self.ctx, self.grad)

    for t, g in zip(self.ctx.parents, grads):
      t.grad = g
      t.backward()

def register(name, cls):
  if name in ['add', 'mul', 'matmul', 'pow']:
    setattr(Tensor, f'__{name}__', lambda *args: cls.apply(cls, *args))
  else: 
    setattr(Tensor, f'{name}', lambda *args: cls.apply(cls, *args))

import synapse.operations
[register(name.lower(), cls) for name, cls in inspect.getmembers(synapse.operations, inspect.isclass)]
