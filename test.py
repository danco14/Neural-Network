import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

test1 = np.arange(50).reshape(10,5)
test2 = np.arange(10)
test2 = np.array([4, 6, ])

x = np.random.permutation(test1.shape[0])

print(test1[x])
print(test2[x])

b = 2

# for n in range(10 // b):
#     rand1 = test1[x[n*b:(n+1)*b]]
#     rand2 = test2[x[n*b:(n+1)*b]]
#     print(rand1)
#     print(rand2)

rand1 = test1[x]
rand2 = test2[x]

print(np.sum(np.argmax(test1) == test2))
