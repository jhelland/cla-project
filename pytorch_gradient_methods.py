
import numpy as np
import theano
from utils import *

import torch
from torch.optim import Optimizer
from functools import reduce


class SGD(Optimizer):

    def __init__(self, params, lr, momentum=0, dampening=0):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening)
        super(SGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        perform a single step of stochastic gradient descent
        
        :param closure: (optional) callable function that reevaluates the model
        and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            dampening = group['dampening']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    d_p = buf

                p.data.add_(-group['lr'] * d_p)

        return loss


class BFGS(Optimizer):

    def __init__(self, params, grad_tol=1e-5, hessian_init=None, lr=None):
        defaults = dict(grad_tol=grad_tol, hessian_init=hessian_init, lr=lr)
        super(BFGS, self).__init__(params, defaults)

        if hessian_init is None:
            numel = 0
            for p in self.param_groups[0]['params']:
                numel += p.numel()
            self.hessians = [torch.eye(numel, numel)]
        else:
            self.hessians = [hessian_init]
        self.alphas = []
        self._params = self.param_groups[0]['params']
        self._numel_cache = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(),
                                       self._params,
                                       0)
        return self._numel_cache

    def _flatten_grad(self):
        """
        compute vectorized version of the parameters (it's like deja-vu)
        """
        views = []
        for p in self.param_groups[0]['params']:
            if p.grad is None:  # fill that bad boy with zeros
                view = p.data.new(p.data.numel()).zero_()
            else:
                view = p.grad.data.view(-1)
            views.append(view)

        return torch.cat(views, 0)

    def _flatten_params(self):
        """
        compute vectorized version of the model parameters
        """
        views = []
        for p in self.param_groups[0]['params']:
            view = p.data.view(-1)
            views.append(view)

        return torch.cat(views, 0)

    def _alter_params(self, update):
        """
        replace the current model parameters with a new set
        
        :param update: (vector) new set of parameters
        """
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.data.mul_(0).add_(update[offset:offset + numel])
            offset += numel
        assert offset == self._numel()

    def weak_wolfe_line_search(self, phi_fn, phi_grad_fn, c1=1e-4, c2=0.9):
        """
        Weak Wolfe conditions (note that 0 < c_1 < c_2 < 1):
            i) (Armijo condition) \phi(\alpha_j) \leq \phi(0) + \alpha_j*c_1*\phi'(0)
            ii) (Weak Wolfe condition) \phi'(\alpha_j) \geq c_2*\phi'(0)
        :param phi_fn: \phi(\alpha) = f(x^{(j)} + \alpha*p^{(j)}), \alpha \geq 0
        :param phi_grad_fn: \phi'(\alpha) = f'(.)*p^{(j)}
        """
        alpha = 1
        mu = 0
        v = np.Inf
        while True:
            if phi_fn(alpha) > phi_fn(0) + alpha * c1 * phi_grad_fn(0):  # Armijo fails
                v = alpha
            elif phi_grad_fn(alpha) < c2 * phi_grad_fn(0):  # Weak Wolfe fails
                mu = alpha
            else:  # both conditions hold so stop
                break

            if v < np.Inf:
                alpha = (mu + v) / 2.
            else:
                alpha *= 2
        return alpha

    def step(self, closure, loss):
        """
        Take a step forward in the training process
        
        :param closure: (function) reevaluates the model given current params
        :param loss: (function) reevaluates loss/cost given current params
        """
        # Compute a search direction p
        B = self.hessians[-1]
        flat_grad = self._flatten_grad()
        B_LU, B_LU_pivots = torch.btrifact(B.unsqueeze(0))  # LU factorization w/ pivoting of B
        grad = flat_grad.neg().unsqueeze(0)  # format gradient
        p = grad.btrisolve(B_LU, B_LU_pivots).view(-1)  # linear system solve

        # Compute appropriate step size using line search in direction p
        flat_params = self._flatten_params()
        def phi_fn(new_params):
            self._alter_params(flat_params + new_params*p)  # insert new params
            ret = loss().data.numpy()
            self._alter_params(flat_params)  # go back to the old params
            return ret

        def phi_grad_fn(new_params):
            self._alter_params(flat_params + new_params*p)
            self.zero_grad()
            loss().backward()
            ret = p.dot(self._flatten_grad())
            self._alter_params(flat_params)
            return ret

        alpha = self.weak_wolfe_line_search(phi_fn, phi_grad_fn)

        # update parameters and Hessian approximation
        s = alpha * p
        flat_params_new = flat_params + s
        self.zero_grad()
        self._alter_params(flat_params_new)
        loss().backward()
        y = self._flatten_grad() - flat_grad

        self.hessians.append(  # note: ger(.) outer product
            B + y.ger(y)/y.dot(s) - (B @ s.ger(s) @ B)/s.dot(B @ s)
        )
