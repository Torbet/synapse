from synapse.tensor import Tensor
import torch

a = Tensor([1.,2.,3.])
b = Tensor([4.,5.,6.])
c = Tensor([7.,8.,9.])

d = a ** b
e = d @ c

e.backward()

print(a.data, a.grad)
print(b.data, b.grad)
print(c.data, c.grad)

a = torch.tensor([1.,2.,3.], requires_grad=True)
b = torch.tensor([4.,5.,6.], requires_grad=True)
c = torch.tensor([7.,8.,9.], requires_grad=True)

d = a ** b
e = d @ c

e.backward()

print(a.data, a.grad)
print(b.data, b.grad)
print(c.data, c.grad)
