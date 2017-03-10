
import numpy as np


def gradient_descent(starting_x, grad_fn, learning_rate, epochs):
    """
    Standard gradient descent algorithm.
    :param starting_x: initial guess for parameter values
    :param grad_fn: callable function returning gradient evaluated at parameter
    :param learning_rate: how quickly to learn. If this argument is too large, the algorithm will overshoot the minimum.
    :param epochs: how many times to run the algorithm
    :return:
    """

    x = starting_x
    for i in range(epochs):
        x -= learning_rate * grad_fn(x)
    return x


def stochastic_gradient_descent(observations, responses, starting_x, grad_fn, learning_rate, epochs, batch_size=1):
    """
    Stochastic gradient descent (SGD) algorithm with batch implementation (BSGD).
    :param observations:
    :param responses:
    :param starting_x: starting guess for the parameters
    :param grad_fn: callable function that returns the gradient wrt the parameters
    :param learning_rate:
    :param epochs: how many iterations of stochastic gradient descent to perform
    :param batch_size: the number of distinct equally sized sets to split the data into
    :return: parameter values after epochs of SGD
    """

    m, n = observations.shape
    x = starting_x
    data_temp = np.concatenate([observations, responses.reshape(m, 1)], axis=1)
    for e in range(epochs):
        print('...Epoch %i/%i' % (e, epochs))

        np.random.shuffle(data_temp)
        batches = np.split(data_temp, batch_size)
        for batch in batches:
            x -= learning_rate * grad_fn(
                batch[:, 0:n].reshape(m // batch_size, n),
                batch[:, -1].reshape(m // batch_size, ),
                x
            )[0]
    return x