from synapse.tensor import Tensor

class Function:
  def __init__(self, *tensors):
    self.parents = tensors
    self.saved = []

  def save(self, *tensors): 
    self.saved.extend(tensors)

  def apply(self, *args):
    ctx = self(*args)
    ret = Tensor(ctx.forward(ctx, *[t.data for t in args]))
    ret.ctx = ctx
    return ret
