
# 04.03.2020

import torch
import torch.nn as nn
import torch.optim as optim

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

W = torch.zeros(1)
W = torch.zeros(1, requires_grad=True)

#lr = 0.1

optimizer = optim.SGD([W], lr=0.15)


nb_epochs = 10

for epoch in range(nb_epochs + 1):
    hypothesis = x_train * W

    cost = torch.mean((hypothesis - y_train) ** 2)
    #gradient = torch.sum((W*x_train-y_train) * x_train)

    print('Epoch {:4d}/{} W: {:.3f}, Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), cost.item()
        ))
    #W -= lr * gradient

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()


