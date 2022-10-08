from synapse.tensor import Tensor

a = Tensor([1,2,3])
b = Tensor([4,5,6])

c = a + b

c.backward()
print("data: ", a.data, "grad: ", a.grad)
print("data: ", b.data, "grad: ", b.grad)
print("data: ", c.data, "grad: ", c.grad)
