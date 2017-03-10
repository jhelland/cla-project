
#################################
# LIBRARIES
#################################

import matplotlib.pyplot as plt
import numpy as np
import seaborn

from mpl_toolkits.mplot3d import Axes3D


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
        np.linspace(0, 10, n_obs) + np.random.normal(0, 1, n_obs)
        for _ in range(n)
    ])
    response = response_fn(x) + np.random.normal(0, 1, n_obs)
    data_matrix = np.vstack((x, np.ones(n_obs))).T
    return data_matrix, response


def plot_regression(x_data, response, beta, plot_type='2d', title=''):
    """
    Simple plotting function given output of linear regression.
    :param x_data: observations
    :param response: response data
    :param coeff: linear regression coefficients
    :param plot_type:
    :return:
    """

    fig = plt.figure(figsize=(8, 8))

    if plot_type is '2d':
        x_range = (min(x_data), max(x_data))
        x = np.arange(x_range[0], x_range[1], (x_range[1] - x_range[0])//2)
        y = beta[0]*x + beta[1]

        ax = fig.add_subplot(111)
        ax.scatter(x_data, response, color='r')
        ax.plot(x, y, linewidth=2, color='k')
        ax.set_title(title)

    elif plot_type is '3d':
        xrange = (min(x_data[:, 0]), max(x_data[:, 0]))
        yrange = (min(x_data[:, 1]), max(x_data[:, 1]))
        xx, yy = np.meshgrid(np.arange(xrange[0], xrange[1], (xrange[1] - xrange[0]) // 2),
                             np.arange(yrange[0], yrange[1], (yrange[1] - yrange[0]) // 2))
        zz = beta[0] * xx + beta[1] * yy + beta[2]

        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_data[:, 0], x_data[:, 1], response, color='r')
        ax.plot_surface(xx, yy, zz,
                        rstride=1, cstride=1,
                        color='None', alpha=0.4)
        ax.set_title(title)
        ax.view_init(10, -50)
    else:
        print('ERROR: not implemented')
    plt.show()


def cost_Function(A, b, x):
    return 1 / float(len(b)) * np.linalg.norm(np.dot(A, x) - b) ** 2


def step_Gradient(A, b, x, learning_Rate):
    N = float(len(b))
    x_Gradient = 2 / N * np.dot(np.transpose(A), (np.dot(A, x) - b))
    new_x = x - learning_Rate * x_Gradient
    return new_x


def gradient_Descent_Runner(A, b, starting_x, learning_Rate, num_Iterations):
    x = starting_x
    for i in range(num_Iterations):
        x = step_Gradient(A, b, x, learning_Rate)
    return x


def linear_f(x, t):
    return x[0] * t + x[1]


def run(learning_rate=0.001, num_iterations=5):

    data, response = generate_data(
        lambda x: 2.0 * sum(xi for xi in x) + 1.0,
        n=2,
        n_obs=100
    )
    beta = np.zeros(3)
    print(beta)

    x_range = (min(data[:, 0]), max(data[:, 0]))
    t = np.linspace(x_range[0], x_range[1], (x_range[1] - x_range[0])//2)

    plot_regression(data[:, 0:2], response, beta, plot_type='3d', title='Gradient descent iteration 0')

    print('Starting gradient descent at error = %f' % cost_Function(data, response, beta))
    print('Running...')

    for i in range(num_iterations):
        beta = gradient_Descent_Runner(data, response, beta, learning_rate, num_iterations)
        plot_regression(data[:, 0:2], response, beta, plot_type='3d', title=('Gradient descent iteration %i' % (i+1)))

    print("After %i iterations: \t beta = %s, \t error = %f"
          % (num_iterations, str(beta), cost_Function(data, response, beta)))


#################################
# MAIN SCRIPT
#################################

run(learning_rate=0.001, num_iterations=5)
