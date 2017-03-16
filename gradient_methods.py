
import numpy as np
import theano
from utils import *


def gradient_descent(theta_init, grad_fn, learning_rate, epochs):
    """
    Standard gradient descent algorithm.
    :param starting_x: initial guess for parameter values
    :param grad_fn: callable function returning gradient evaluated at parameter
    :param learning_rate: how quickly to learn. If this argument is too large, the algorithm will overshoot the minimum.
    :param epochs: how many times to run the algorithm
    :return:
    """

    theta = theta_init
    for i in range(epochs):
        theta = ([a - learning_rate*b for a, b in zip(theta, grad_fn(theta))]
                 if isinstance(theta, list) else theta - learning_rate*grad_fn(theta))
    return theta


def stochastic_gradient_descent(observations,
                                responses,
                                theta_init,
                                grad_fn,
                                learning_rate,
                                epochs,
                                batch_size=1):
    """
    Stochastic gradient descent (SGD) algorithm with batch implementation (BSGD).
    :param observations:
    :param responses:
    :param theta_init: starting values for the model parameters
    :param grad_fn: callable function that returns the gradient wrt the parameters
    :param learning_rate:
    :param epochs: how many iterations of stochastic gradient descent to perform
    :param batch_size: the number of distinct equally sized sets to split the data into
    :return: parameter values after epochs of SGD
    """

    if len(observations.shape) > 1:
        m, n = observations.shape
    else:
        m, n = observations.shape[0], 1
        observations = observations.reshape(m, n)

    theta = theta_init
    data_temp = np.concatenate([observations, responses.reshape(m, 1)], axis=1)
    for e in range(epochs):

        np.random.shuffle(data_temp)
        batches = np.split(data_temp, batch_size)
        for batch in batches:
            dx = [learning_rate * entry for entry in grad_fn(
                batch[:, 0:n].reshape((m // batch_size, n) if n > 1 else (m // batch_size)),
                batch[:, -1].reshape(m // batch_size, ),
                theta
            )]
            if len(dx) == 1:
                theta -= dx[0]
            else:
                theta = [a - b for a, b in zip(theta, dx)]
    return theta


def gauss_newton(jacobian,
                 residuals,
                 theta_init,
                 max_iterations,
                 tol=1e-6,
                 format_rule=lambda arg: arg):

    theta = theta_init
    i = 0
    while (i < max_iterations):
        i += 1

        J = jacobian(theta)
        H = J.T @ J
        resid = residuals(theta)
        update = np.linalg.solve(H, -J.T @ resid)
        theta = vec(theta, axis=1) + update

        if format_rule is not None:
            theta = format_rule(theta)

    return theta


def levenberg_marquardt(jacobian,
                        residuals,
                        cost,
                        theta_init,
                        max_iterations,
                        tau=0.1,
                        adjustment_factor=10.0,
                        tol=1e-6,
                        format_rule=None,
                        W_init=None):

    theta = theta_init
    I = np.identity(vec(theta).size)
    _tau = np.zeros(max_iterations + 1)
    _tau[0] = tau
    i = 0
    while (i < max_iterations):# and (dx[dx > tol].size > 0):
        i += 1

        resid = residuals(theta)
        J = jacobian(theta)
        H = J.T @ J
        g = J.T @ resid
        try:
            update = np.linalg.solve(
                H + tau * I,
                g
            )
        except np.linalg.LinAlgError:
            print('ERROR: singular matrix')
            tau = 1.0
            continue
            #return theta, _tau

        theta_new = ([a - b for a, b in zip(theta, format_rule(update))] if isinstance(theta, list)
                     else theta - update)

        if cost(theta) < cost(theta_new):
            tau *= adjustment_factor
            adjustment_factor *= 2
        else:
            theta = theta_new
            tau *= 1/3
            adjustment_factor = 2.0

        _tau[i] = tau

    return theta, _tau
