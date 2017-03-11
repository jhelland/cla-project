
#################################
# LIBRARIES
#################################

# personal files
from gradient_methods import *

# plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn
from mpl_toolkits.mplot3d import Axes3D

# computation & data structures
import numpy as np
import theano
import theano.tensor as T
import math

# misc
from copy import deepcopy


#################################
# FUNCTIONS
#################################

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


def response_fn1(x):
    return x[0] * np.sin(x[0])


def run(n_obs=100, n_iterations=5):
    observations, response = generate_data(response_fn1, n=1, n_obs=n_obs)
    observations = observations[:, 0]

    n_eval = 100
    x = np.linspace(-5, 5, n_eval)

    print('...Initializing model')
    X = T.vector('X')
    y = T.vector('y')
    W1 = T.matrix('W1')
    W2 = T.matrix('W2')
    b1 = T.vector('b1')
    b2 = T.vector('b2')
    theta = [W1, b1, W2, b2]

    model = T.dot(W2, T.nnet.sigmoid(T.dot(W1, X) + b1)) + b2
    cost = (1 / n_obs) * T.sum(
        (y - model)**2
    )
    cost_grad = T.grad(cost=cost, wrt=theta)
    cost_fn = theano.function(inputs=theta,
                              outputs=cost,
                              givens={
                                  X: observations,
                                  y: response
                              })
    cost_grad_fn = theano.function(inputs=[X, y] + theta,
                                   outputs=cost_grad)

    low = -5.0
    high = 5.0
    theta_init = [np.random.uniform(low, high, size=(10, 1)),
                  np.random.uniform(low, high, size=(10, )),
                  np.random.uniform(low, high, size=(1, 10)),
                  np.random.uniform(low, high, size=(1, ))]

    for i in range(n_iterations):
        print('iteration %i/%i' % (i, n_iterations))
        theta_init = stochastic_gradient_descent(observations,
                                                 response,
                                                 theta_init,
                                                 lambda a, b, c: cost_grad_fn(a.T, b, *c),
                                                 learning_rate=0.005,
                                                 epochs=100,
                                                 batch_size=n_obs)

    f = (
        T.dot(theta_init[2],
              T.nnet.sigmoid(T.dot(theta_init[0], x.reshape(1, n_eval)) + theta_init[1].reshape(theta_init[1].shape[0], 1)))
        + theta_init[3].reshape(theta_init[3].shape[0], 1)
    ).eval()

    fig = plt.figure(figsize=plt.figaspect(0.5))
    plt.axis('off')
    ax = fig.add_subplot(121)
    ax.scatter(observations, response, color=seaborn.color_palette()[2])
    ax.plot(x, response_fn1([x]), '-', color=seaborn.color_palette()[0], linewidth=2.5)
    ax.set_title('Original Process')

    ax = fig.add_subplot(122)
    ax.scatter(observations, response, color=seaborn.color_palette()[2])
    ax.plot(x, f.flatten())
    ax.set_title('Fitted Model')
    plt.show()


#################################
# MAIN
#################################

if __name__ == '__main__':
    run(n_iterations=100, n_obs=20)
