from synapse.function import Function

class Add(Function):
  @staticmethod
  def forward(ctx, a, b):
    ctx.save(a, b)
    return a+b
  
  @staticmethod
  def backward(ctx, grad):
    return grad, grad

class Mul(Function):
  @staticmethod
  def forward(ctx, a, b):
    ctx.save(a, b)
    return a*b
  
  @staticmethod
  def backward(ctx, grad):
    a,b = ctx.saved
    return b*grad, a*grad

class MatMul(Function):
  @staticmethod
  def forward(ctx, a, b):
    ctx.save(a, b)
    return a@b
  
  @staticmethod
  def backward(ctx, grad):
    a,b = ctx.saved
    """
    ga = grad @ b.T
    gb = a.T @ grad
    """
    return b*grad, a*grad

