from synapse.function import Function
import numpy as np

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

class Pow(Function):
  @staticmethod
  def forward(ctx, a, b):
    out = a ** b
    ctx.save(a, b, out)
    return a ** b
  
  @staticmethod
  def backward(ctx, grad):
    a,b,out = ctx.saved
    ga = b * a ** (b-1) * grad
    gb = out * np.log(a) * grad
    return ga, gb

class Sum(Function):
  @staticmethod
  def forward(ctx, a):
    ctx.save(a)
    return a.sum()
  
  @staticmethod
  def backward(ctx, grad):
    a = ctx.saved[0]
    return np.ones(a.shape) * grad

class ReLU(Function):
  @staticmethod
  def forward(ctx, a):
    ctx.save(a)
    return np.maximum(a, 0)
  
  @staticmethod
  def backward(ctx, grad):
    a = ctx.saved[0]
    return (a >= 0) * grad
