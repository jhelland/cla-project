"""
nonlinear_regression.py ported to PyTorch library
TODO:
- get existing optimization methods working here
"""

import torch
import torch.autograd
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import seaborn

from pytorch_gradient_methods import *


def response_fn1(x):
    return x[0] * np.sin(x[0])


def generate_data(response_fn, n=1, n_obs=100):
    """
    Generate data according to a function with Gaussian noise in n dimensions.
    Currently adds noise to the inputs as well as outputs.
    :param response_fn:
    :param n: dimension of data
    :param n_obs: number of observations
    :return:
    """

    x = np.array([
        np.linspace(-5, 5, n_obs)
        for _ in range(n)
    ])
    response = response_fn(x) + np.random.normal(0, 0.1, n_obs)
    data_matrix = np.vstack((x, np.ones(n_obs))).T
    return data_matrix, response


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = nn.Linear(1, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, x):
        """
        forward propogate input through the neural network
        :param x: 
        :return: 
        """
        x = F.tanh(self.hidden(x))
        x = self.output(x)
        return x


if __name__ == '__main__':

    n_obs = 20
    observations, response = generate_data(response_fn1, n=1, n_obs=n_obs)
    observations = observations[:, 0].reshape(observations.shape[0], 1)

    model = Net()
    model.train()

    # optimizer = SGD(model.parameters(), lr=0.005, momentum=0.5, dampening=0)
    optimizer = BFGS(model.parameters(), grad_tol=1e-5)

    model.train()  # puts the model in training mode
    learning_rate = 0.005
    for i in range(100):
        data = Variable(torch.Tensor(observations))
        target = Variable(torch.Tensor(response))

        optimizer.zero_grad()
        output = model(data)

        def closure():
            return model(data)

        def loss():
            return (closure() - target).pow(2).sum() / n_obs

        if i % 10 == 0:
            print(i, loss().data[0])

            x = np.linspace(-5, 5, 1000).reshape(1000, 1)
            x = Variable(torch.Tensor(x))
            y = model(x)

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
            ax.scatter(data.data.numpy(), target.data.numpy())
            ax.plot(x.data.numpy(), y.data.numpy(), 'k-')
            plt.show()
            plt.close()

        loss().backward()  # back propogate through the neural network
        optimizer.step(closure, loss)
