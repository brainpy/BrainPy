import torch.nn


b = torch.nn.BatchNorm1d(3)

l = torch.nn.LayerNorm([3, 4], elementwise_affine=False)

inputs = torch.ones((2, 3, 4))
inputs[0, 0, :] = 2.
inputs[0, 1, 0] = 5.
print(inputs)

print(l(inputs))

print(l.bias)
print(l.weight)
