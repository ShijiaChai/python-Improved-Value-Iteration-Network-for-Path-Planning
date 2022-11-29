import torch as t
from torch.autograd import Variable
a = t.tensor([[1.0 ,2.0],[3.0, 4.0]])
b = t.tensor([5.0,6.0]).reshape(-1, 1)
print(a)
c = a + b
print(c)
