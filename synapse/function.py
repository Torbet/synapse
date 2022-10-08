from synapse.tensor import Tensor

class Function:
  def __init__(self, *tensors):
    self.parents = tensors[1]
    self.saved = []

  def save(self, *tensors): 
    self.saved.extend(tensors)

  @classmethod
  def apply(cls, *args):
    ctx = cls(*args)
    ret = Tensor(ctx.forward(ctx, *[t.data for t in args[1]]))
    ret.ctx = ctx
    return ret
