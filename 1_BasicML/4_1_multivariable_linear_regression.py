# 04.03.2020

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    def forward(self, x):
        return self.linear(x)

class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)
    def forward(self, x):
        return self.linear(x)
    



x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

#W = torch.zeros((3, 1), requires_grad=True)
#b = torch.zeros(1, requires_grad=True)

model = MultivariateLinearRegressionModel()

#optimizer = optim.SGD([W, b], lr=1e-5)
optimizer = optim.SGD(model.parameters(), lr=1e-5)


nb_epochs = 20
for epoch in range(nb_epochs + 1):
    #hypothesis = x_train.matmul(W) + b
    #cost = torch.mean((hypothesis - y_train) ** 2)
    prediction = model(x_train)
    cost = F.mse_loss(prediction, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} Cost : {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
