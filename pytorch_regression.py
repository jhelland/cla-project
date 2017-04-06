
from itertools import count

import torch
import torch.autograd
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import seaborn


N = 64

alpha = 1.3
beta = np.array([[0.5]])

x_data = np.random.randn(N, 1)
y_data = x_data.dot(beta) + alpha + np.random.normal(0.0, 0.1, (N, 1))

x = Variable(torch.Tensor(x_data), requires_grad=False)
y = Variable(torch.Tensor(y_data), requires_grad=False)

w_beta = Variable(torch.randn(1, 1), requires_grad=True)
w_alpha = Variable(torch.randn(1), requires_grad=True)

learning_rate = 0.01
optimizer = torch.optim.SGD([w_beta, w_alpha], lr=learning_rate)

def cost(input, target, w_beta, w_alpha):
    return (input.mm(w_beta).add(w_alpha.expand(N)) - target).pow(2).sum()

def cost_uni(input, target, w_beta, w_alpha):
    return (input.mm(w_beta).add(w_alpha) - target).pow(2).sum()

def output(input, w_beta, w_alpha):
    return input.mm(w_beta).add(w_alpha.expand(N))

w_path = None
for i in range(100):

    if w_path is None:
        w_path = [[w_beta.data.numpy().flatten()[0], w_alpha.data.numpy().flatten()[0]]]
    else:
        w_path = np.vstack([w_path, [w_beta.data.numpy().flatten()[0], w_alpha.data.numpy().flatten()[0]]])

    cost_i = cost(x, y, w_beta, w_alpha)

    if i % 10 == 0:
        print(i, cost_i.data[0])

        pred = output(x, w_beta, w_alpha).data.numpy()

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.scatter(x.data.numpy(), y.data.numpy())
        ax.plot(x.data.numpy(), pred, 'k-')
        plt.show()
        plt.close()

    optimizer.zero_grad()

    cost_i.backward(retain_variables=True)

    optimizer.step()
