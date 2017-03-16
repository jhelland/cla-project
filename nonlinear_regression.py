
#################################
# LIBRARIES
#################################

# personal files
from gradient_methods import *
from utils import *

# graphics
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn
import imageio
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

def make_gif(fin_dir, fin_name, fout_name, n_images, duration=5):
    images = []
    for i in range(n_images):
        images.append(imageio.imread((fin_dir + fin_name + '%i.png') % i))
    kargs = {'duration': duration}
    imageio.mimsave(fout_name, images, 'GIF', **kargs)


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


def get_grid(xlim, ylim, step=1.0):
    return [np.arange(xlim[0], xlim[1], step=step),
            np.arange(ylim[0], ylim[1], step=step)]


def response_fn1(x):
    return x[0] * np.sin(x[0])


def response_fn2(x):
    return np.sin(x[0]) + np.cos(x[1])


def run2(n_obs=5, method='GD', n_iterations=1, image_rate=1, **kwargs):

    kwargs_keys = list(kwargs.keys())
    if 'epochs' in kwargs_keys:
        epochs = kwargs['epochs']
    else:
        epochs = 1
    if 'tol' in kwargs_keys:
        tol = kwargs['tol']
    else:
        tol = 1e-10
    if 'learning_rate' in kwargs_keys:
        learning_rate = kwargs['learning_rate']
    else:
        learning_rate = 0.1
    if 'batch_size' in kwargs_keys:
        batch_size = kwargs['batch_size']
    else:
        batch_size = n_obs
    method = str.upper(method)

    seaborn.set_style('white')

    xx, yy = np.meshgrid(np.linspace(-5.0, 5.0, n_obs), np.linspace(-5.0, 5.0, n_obs))
    observations = np.vstack([xx.flatten(), yy.flatten()])
    response = response_fn2(observations) + np.random.normal(0, 0.2, n_obs**2)

    print('...Initializing model')
    X = T.matrix('X')
    y = T.vector('y')
    W1 = T.matrix('W1')
    W2 = T.matrix('W2')
    b1 = T.TensorType(dtype=theano.config.floatX,
                      broadcastable=(False, True))('b1')
    b2 = T.TensorType(dtype=theano.config.floatX,
                      broadcastable=(False, True))('b2')
    theta = [W1, b1, W2, b2]
    low = -5.0
    high = 5.0
    theta_init = [np.random.uniform(low, high, size=(20, 2)),
                  np.random.uniform(low, high, size=(20, 1)),
                  np.random.uniform(low, high, size=(1, 20)),
                  np.random.uniform(low, high, size=(1, 1))]
    theta_init = [entry.astype(theano.config.floatX) for entry in theta_init]
    theta_dict = {
        'GD': theta_init,
        'SGD': theta_init,
        'GN': theta_init,
        'LM': theta_init
    }

    model = T.dot(W2, T.nnet.sigmoid(T.dot(W1, X) + b1)) + b2

    model_fn = theano.function(inputs=[X] + theta,
                               outputs=model)

    cost = (1 / n_obs) * T.sum(
        (y - model) ** 2
    )
    cost_fn = theano.function(inputs=theta,
                              outputs=cost,
                              givens={
                                  X: observations,
                                  y: response
                              })
    cost_grad = T.grad(cost=cost, wrt=theta)
    cost_grad_fn = theano.function(inputs=[X, y] + theta,
                                   outputs=cost_grad)

    # functions for use with Gauss-Newton and Levenberg-Marquardt algorithms
    n_params = vec(theta_init).size

    def jacobian(args):
        return vec(cost_grad_fn(observations, response, *args), axis=1).reshape(n_params, 1)

    def format_rule(arg):
        return format_params(arg.flatten(), [a.shape for a in theta_init])

    # train
    method_names = {'GD': 'Gradient Descent, learning rate %04.2f' % learning_rate,
                    'SGD': 'Stochastic Gradient Descent, learning rate %04.2f' % learning_rate,
                    'GN': 'Gauss-Newton',
                    'LM': r'Levenberg-Marquardt, $\lambda_0 = %04.2f$, $v = %04.2f$' % (0.1, 10.0)}
    # print('\t...Running method with %s' % method_names[str.upper(method)])
    n_images = [0, 0]
    fout_name_template = r'./gif_figures/iteration{}.png'
    fig = []
    lambda_i = 0.1
    angle = 0
    for i in range(n_iterations):

        if method == 'GD' or method == 'ALL':
            def grad_fn(arg):
                return cost_grad_fn(observations.reshape(2, n_obs**2), response, *arg)

            theta_dict['GD'] = gradient_descent(theta_init=theta_dict['GD'],
                                                grad_fn=grad_fn,
                                                learning_rate=learning_rate,
                                                epochs=epochs)
        if method == 'SGD' or method == 'ALL':
            def grad_fn(a, b, c):
                return cost_grad_fn(a.reshape(2, n_obs // batch_size), b, *c)

            theta_dict['SGD'] = stochastic_gradient_descent(observations=observations.T,
                                                            responses=response,
                                                            theta_init=theta_dict['SGD'],
                                                            grad_fn=grad_fn,
                                                            learning_rate=learning_rate,
                                                            epochs=epochs,
                                                            batch_size=batch_size)
        if method == 'GN' or method == 'ALL':
            theta_dict['GN'] = gauss_newton(jacobian=jacobian,
                                            residuals=lambda arg: cost_fn(*arg),
                                            theta_init=theta_dict['GN'],
                                            max_iterations=1,
                                            tol=tol,
                                            format_rule=format_rule)
        if method == 'LM' or method == 'ALL':
            theta_dict['LM'], lambda_i = levenberg_marquardt(jacobian=jacobian,
                                                             residuals=lambda arg: cost_fn(*arg),
                                                             theta_init=theta_dict['LM'],
                                                             max_iterations=epochs,
                                                             lambda_init=lambda_i,
                                                             tol=tol,
                                                             format_rule=format_rule)
        """
        if i % image_rate == 0:

            xxx, yyy = np.meshgrid(np.linspace(-5.0, 5.0, 20), np.linspace(-5.0, 5.0, 20))
            zzz = response_fn2(np.vstack([xxx.flatten(), yyy.flatten()])).reshape(xxx.shape)

            f = model_fn(np.vstack([xxx.flatten(), yyy.flatten()]),
                         *theta_dict[method]).reshape(xxx.shape)

            fig = plt.figure(figsize=plt.figaspect(0.5))
            plt.axis('off')
            ax = fig.add_subplot(121, projection='3d')
            ax.plot_surface(xxx, yyy, zzz,
                            color='None', alpha=0.4)
            ax.scatter(xx, yy, response.reshape(xx.shape),
                       color=seaborn.color_palette()[2], s=100, alpha=1.0)
            ax.set_title('Original Process')
            ax.set_xlabel(r'$x_1$')
            ax.set_ylabel(r'$x_2$')
            ax.set_zlabel(r'$f(X) = x_1^2 - x_2^2$')
            ax.view_init(30, angle)

            ax = fig.add_subplot(122, projection='3d')
            ax.scatter(xx, yy, response.reshape(xx.shape),
                       color=seaborn.color_palette()[2], s=100, alpha=1.0)
            ax.plot_surface(xxx, yyy, f.reshape(xxx.shape),
                            color='None', alpha=0.4)
            ax.set_title('Fitted Model')
            ax.set_xlabel(r'$x_1$')
            ax.set_ylabel(r'$x_2$')
            ax.set_zlabel(r'$M(X) = W_2\Psi(W_1X + b_1) + b_2$')
            ax.view_init(30, angle)

            n_images[0] += 1
            angle += 1

        if i % (20 * image_rate) == 0:
            print('iteration %i/%i' % (i, n_iterations))
            for fig_num in plt.get_fignums():
                plt.figure(fig_num)
                plt.savefig('./gif_figures/iteration%i.png' % n_images[1], bbox_inches='tight')
                plt.close()
                n_images[1] += 1

    for fig_num in plt.get_fignums():
        plt.figure(fig_num)
        plt.savefig('./gif_figures/iteration%i.png' % n_images[1], bbox_inches='tight')
        plt.close()
        n_images[1] += 1

    make_gif(fin_dir='./gif_figures/',
             fin_name='iteration',
             fout_name='./nnet_training.gif',
             n_images=n_images[0],
             duration=0.1)

    """
    xxx, yyy = np.meshgrid(np.linspace(-5.0, 5.0, 20), np.linspace(-5.0, 5.0, 20))
    zzz = response_fn2(np.vstack([xxx.flatten(), yyy.flatten()])).reshape(xxx.shape)

    f = model_fn(np.vstack([xxx.flatten(), yyy.flatten()]),
                 *theta_dict[method]).reshape(xxx.shape)

    fig = plt.figure(figsize=plt.figaspect(0.5))
    plt.axis('off')
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(xxx, yyy, zzz,
                    color='None', alpha=0.4)
    ax.scatter(xx, yy, response.reshape(xx.shape),
               color=seaborn.color_palette()[2], s=100, alpha=1.0)

    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(xx, yy, response.reshape(xx.shape),
               color=seaborn.color_palette()[2], s=100, alpha=1.0)
    ax.plot_surface(xxx, yyy, f.reshape(xxx.shape),
                    color='None', alpha=0.4)
    ax.set_title('Fitted Model')

    plt.show()



def run(n_obs=100, n_iterations=5, image_rate=1, method='SGD', **kwargs):

    kwargs_keys = list(kwargs.keys())
    if 'epochs' in kwargs_keys:
        epochs = kwargs['epochs']
    else:
        epochs = 1
    if 'tol' in kwargs_keys:
        tol = kwargs['tol']
    else:
        tol = 1e-10
    if 'learning_rate' in kwargs_keys:
        learning_rate = kwargs['learning_rate']
    else:
        learning_rate = 0.1
    if 'batch_size' in kwargs_keys:
        batch_size = kwargs['batch_size']
    else:
        batch_size = n_obs
    method = str.upper(method)

    seaborn.set_style('white')

    observations, response = generate_data(response_fn1, n=1, n_obs=n_obs)
    observations = observations[:, 0].reshape(1, observations.shape[0])

    observations = observations.astype(theano.config.floatX)
    response = response.astype(theano.config.floatX)

    print('...Initializing model')
    X = T.matrix('X')
    y = T.vector('y')
    W1 = T.matrix('W1')
    W2 = T.matrix('W2')
    b1 = T.TensorType(dtype=theano.config.floatX,
                      broadcastable=(False, True))('b1')
    b2 = T.TensorType(dtype=theano.config.floatX,
                      broadcastable=(False, True))('b2')
    theta = [W1, b1, W2, b2]

    l_s = [1, 3, 1]
    low = -1.0 / np.sqrt(l_s[1])
    high = 1.0 / np.sqrt(l_s[1])
    theta_init = [np.random.uniform(low, high, size=(l_s[1], 1)),
                  np.random.uniform(low, high, size=(l_s[1], 1)),
                  np.random.uniform(low, high, size=(1, l_s[1])),
                  np.random.uniform(low, high, size=(1, 1))]
    theta_init = [entry.astype(theano.config.floatX) for entry in theta_init]
    theta_dict = {
        'GD': theta_init,
        'SGD': theta_init,
        'GN': theta_init,
        'LM': theta_init
    }

    model = T.dot(W2, T.tanh(T.dot(W1, X) + b1)) + b2

    model_fn = theano.function(inputs=[X] + theta,
                               outputs=model)

    cost = (1 / n_obs) * T.sum(
        (y - model)**2
    )
    cost_fn = theano.function(inputs=theta,
                              outputs=cost,
                              givens={
                                  X: observations,
                                  y: response
                              })
    cost_grad = T.grad(cost=cost, wrt=theta)
    cost_grad_fn = theano.function(inputs=[X, y] + theta,
                                   outputs=cost_grad)

    # functions for SGD
    def grad_fn(a, b, c, n_batches):
        return cost_grad_fn(a.reshape(1, n_obs // n_batches), b, *c)

    # functions for use with Gauss-Newton and Levenberg-Marquardt algorithms
    resid = y - model
    resid_fn = theano.function(inputs=theta,
                               outputs=resid,
                               givens={
                                   X: observations,
                                   y: response
                               })

    jac_r = theano.gradient.jacobian(resid.flatten(), theta)
    jac_r_fn = theano.function(inputs=theta,
                               outputs=jac_r,
                               givens={
                                   X: observations,
                                   y: response
                               })

    n_params = vec(theta_init).size
    def jacobian(args):
        return vec(cost_grad_fn(observations, response, *args), axis=1).reshape(n_params, 1)

    def format_rule(arg):
        return format_params(arg.flatten(), [a.shape for a in theta_init])

    def jac(jac_fn, args):
        return (np.vstack(
            [a.squeeze().T for a in jac_fn(*args)]
        )).T

    # train
    method_names = {'GD': 'Gradient Descent, learning rate %04.2f' % learning_rate,
                    'SGD': 'Stochastic Gradient Descent, learning rate %04.2f' % learning_rate,
                    'GN': 'Gauss-Newton',
                    'LM': r'Levenberg-Marquardt, $\lambda_0 = %04.2f$, $v = %04.2f$' % (0.1, 10.0)}
    #print('\t...Running method with %s' % method_names[str.upper(method)])
    n_images = [0, 0]
    fout_name_template = r'./gif_figures/iteration{}.png'
    fig = []
    taus = [0]
    for i in range(n_iterations):

        if method == 'GD' or method == 'ALL':
            theta_dict['GD'] = stochastic_gradient_descent(
                observations=observations.T,
                responses=response,
                theta_init=theta_dict['GD'],
                grad_fn=lambda *args: grad_fn(*args, n_batches=1),
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=1
            )
        if method == 'SGD' or method == 'ALL':
            theta_dict['SGD'] = stochastic_gradient_descent(
                observations=observations.T,
                responses=response,
                theta_init=theta_dict['SGD'],
                grad_fn=lambda *args: grad_fn(*args, n_batches=batch_size),
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size
            )
        if method == 'GN' or method == 'ALL':
            theta_dict['GN'] = gauss_newton(
                jacobian=lambda arg: jac(jac_r_fn, arg),
                residuals=lambda arg: resid_fn(*arg).T,
                theta_init=theta_dict['GN'],
                max_iterations=epochs,
                tol=tol,
                format_rule=format_rule
            )
        if method == 'LM' or method == 'ALL':
            theta_dict['LM'], taus = levenberg_marquardt(
                jacobian=lambda arg: jac(jac_r_fn, arg),
                residuals=lambda arg: resid_fn(*arg).T,
                cost=lambda arg: cost_fn(*arg),
                theta_init=theta_dict['LM'],
                max_iterations=epochs,
                tau=20.0,
                adjustment_factor=10.0,
                tol=1e-6,
                format_rule=format_rule
            )

        """
        if i % image_rate == 0:
            outputs = [(model_fn(x.reshape(1, x.shape[0]), *theta_dict[key]).flatten(),
                        method_names[key])
                       for key in ('GD', 'SGD', 'GN', 'LM')]

            fig = plt.figure(figsize=(10, 10))
            plt.axis('off')

            manager = plt.get_current_fig_manager()
            manager.resize(*manager.window.maxsize())


            ax = fig.add_subplot(231)
            ax.scatter(observations, response, color=seaborn.color_palette()[2])
            ax.plot(x, response_fn1([x]), '-', color=seaborn.color_palette()[0])
            ax.set_title('Original Process')
            ax.set_ylim([-6.0, 3.0])
            ax.set_ylabel(r'$f(x) = x \sin(x)$')
            ax.set_xlabel(r'x')


            ax_num = 1
            for output in outputs:
                ax = fig.add_subplot(2, 2, ax_num)
                ax.scatter(observations, response, color=seaborn.color_palette()[2])
                ax.plot(x, output[0])
                ax.set_title(output[1])
                ax.set_ylim([-6.0, 3.0])
                ax.set_ylabel('Model')
                ax.set_xlabel(r'x')

                ax_num += 1

            n_images[0] += 1

        if i % (20 * image_rate) == 0:
            print('iteration %i/%i' % (i, n_iterations))
            for fig_num in plt.get_fignums():
                plt.figure(fig_num)
                plt.savefig('./gif_figures/iteration%i.png' % n_images[1], bbox_inches='tight')
                plt.close()
                n_images[1] += 1

    for fig_num in plt.get_fignums():
        plt.figure(fig_num)
        plt.savefig('./gif_figures/iteration%i.png' % n_images[1], bbox_inches='tight')
        plt.close()
        n_images[1] += 1

    make_gif(fin_dir='./gif_figures/',
             fin_name='iteration',
             fout_name='./nnet_training.gif',
             n_images=n_images[0],
             duration=0.1)

    """
    n_eval = 100
    x = np.linspace(-5, 5, n_eval).astype(theano.config.floatX)
    f = model_fn(x.reshape(1, 100), *theta_dict[method]).flatten()

    fig = plt.figure(figsize=plt.figaspect(1/3))
    plt.axis('off')
    ax = fig.add_subplot(131)
    ax.scatter(observations, response, color=seaborn.color_palette()[2])
    ax.plot(x, response_fn1([x]), '-', color=seaborn.color_palette()[0])
    ax.set_title('Original Process')
    ax.set_ylim([-6.0, 3.0])

    ax = fig.add_subplot(132)
    ax.scatter(observations, response, color=seaborn.color_palette()[2])
    ax.plot(x, f)
    ax.set_title('Fitted Model')
    ax.set_ylim([-6.0, 3.0])

    if method == 'LM' or method == 'ALL':
        ax = fig.add_subplot(133)
        ax.semilogy(np.arange(1, taus.size+1), taus)
        ax.set_ylabel(r'$\tau_i$')
        ax.set_xlabel(r'$i$')
        ax.set_title('Update Parameter For Levenberg-Marquardt')

    plt.show()
    #plt.close()


#################################
# MAIN
#################################

if __name__ == '__main__':

    run(n_iterations=1,
        n_obs=20,
        image_rate=20,
        method='GD',
        batch_size=20,
        epochs=500,
        learning_rate=0.01)

    """
    run2(n_obs=5,
         n_iterations=100,
         learning_rate=0.01,
         image_rate=10,
         method='LM')
    """
