# Tensor Manipulation 1
# www.edwith.org/boostcourse-dl-pytorch/
# 03.30.2020

import numpy as np
import torch


##### 1D Array with NumPy
print("1D Array with NumPy")
t_1D = np.array([0., 1., 2., 3., 4., 5., 6.])
print(t_1D)

print('Rank  of t: ', t_1D.ndim)
print('Shape of t: ', t_1D.shape)

print('t[0] t[1] t[-1] = ', t_1D[0], t_1D[1], t_1D[-1]) #Element
print('t[2:5] t[4:-1]  = ', t_1D[2:5], t_1D[4:-1])      #Slicing
print('t[:2] t[3:]     = ', t_1D[:2], t_1D[3:])         #Slicing

print()

##### 2D Array with NumPy
print("2D Array with NumPy")
t_2D = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
print(t_2D)

print('Rank  of t: ', t_2D.ndim)
print('Shape of t: ', t_2D.shape)

print()

##### 1D Array with PyTorch
print("1D Array with PyTorch")
t_1D_pytorch = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print(t_1D_pytorch)

print(t_1D_pytorch.dim())
print(t_1D_pytorch.shape)
print(t_1D_pytorch.size())
print(t_1D_pytorch[0], t_1D_pytorch[1], t_1D_pytorch[-1])
print(t_1D_pytorch[2:5], t_1D_pytorch[4:-1])
print(t_1D_pytorch[:2], t_1D_pytorch[3:])

print()

##### 2D Array with PyTorch
print("2D Array with PyTorch")
t_2D_pytorch = torch.FloatTensor([[1., 2., 3.],
                                  [4., 5., 6.],
                                  [7., 8., 9.],
                                  [10., 11., 12.]
                                 ])
print(t_2D_pytorch)
print(t_2D_pytorch.dim())
print(t_2D_pytorch.size())
print(t_2D_pytorch[:, 1])
print(t_2D_pytorch[:, 1].size())
print(t_2D_pytorch[:, :-1])

print()

##### Broadcasting
print("Broadcasting")
m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])
print(m1 + m2)

m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3]])
print(m1 + m2)

m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3], [4]])
print(m1 + m2)

print()

##### Multiplication vs Matrix Multiplication
print("Multiplication vs Matrix Multiplication")
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print(m1.matmul(m2))
print(m1.mul(m2))

print()

##### Mean
print("Mean")
t = torch.FloatTensor([1, 2])
print(t.mean())
t = torch.LongTensor([1, 2])
try:
    print(t.mean())
except Exception as exc:
    print(exc)

t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
print(t.mean())
print(t.mean(dim=0))
print(t.mean(dim=1))
print(t.mean(dim=-1))

print()

##### Sum
print("Sum")
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
print(t.sum())
print(t.sum(dim=0))
print(t.sum(dim=1))
print(t.sum(dim=-1))

print()

##### Max and Argmax
print("Max and Argmax")
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
print(t.max())
print(t.max(dim=0))
print('Max: ', t.max(dim=0)[0])
print('Argmax: ', t.max(dim=0)[1])

print(t.max(dim=1))
print(t.max(dim=-1))

print()

# Tensor Manipulation 2
# 03.31.2020

##### View
print("View")
t = np.array([[[0, 1, 2],
               [3, 4, 5]],
              [[6, 7, 8],
               [9, 10, 11]]])
ft = torch.FloatTensor(t)
print(ft)
print(ft.shape)

print(ft.view([-1, 3]))
print(ft.view([-1, 3]).shape)

print(ft.view([-1, 1, 3]))
print(ft.view([-1, 1, 3]).shape)

print()

##### Squeeze and Unsqueeze
print("Squeeze and Unsqueeze")
ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape)

print(ft.squeeze())
print(ft.squeeze().shape)

print()

ft = torch.Tensor([0, 1, 2])
print(ft)
print(ft.shape)
print(ft.unsqueeze(0))
print(ft.unsqueeze(0).shape)

print(ft.view(1, -1))
print(ft.view(1, -1).shape)

print(ft.unsqueeze(1))
print(ft.unsqueeze(1).shape)

print(ft.unsqueeze(-1))
print(ft.unsqueeze(-1).shape)

print()

##### Type Casting
print("Type Casting")

lt = torch.LongTensor([1, 2, 3, 4])
print(lt)

print(lt.float())
bt = torch.ByteTensor([True, False, False, True])
print(bt)
print(bt.long())
print(bt.float())

print()

##### Concatenate
print("Concatenate")
x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])

print(torch.cat([x, y], dim=0))
print(torch.cat([x, y], dim=1))

print()

##### Stacking
print("Stacking")
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])

print(torch.stack([x, y, z]))
print(torch.stack([x, y, z], dim=1))
print(torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0))

print()

##### Ones and Zeros
print("Ones and Zeros")
x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(x)

print(torch.ones_like(x))
print(torch.zeros_like(x))

print()

##### In-place Operation
print("In-place Operation")
x = torch.FloatTensor([[1, 2], [3, 4]])
print(x.mul(2.))
print(x)
print(x.mul_(2.))
print(x)



