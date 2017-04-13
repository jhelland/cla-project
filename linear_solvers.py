import torch
import numpy as np
from scipy import linalg
import time

def solve (A,b, arg = 'lu'):
    """
    solve the a linear system or the least sqrs problem with different method specified
    :param A: data matrix torch.tensor.float m x n
    :param b: familiar b torch.tensor.foat m
    :param arg: string specifying type of solving method
    :return x, t: torch.tensor.float, time to compute each method
    """
    t = time.time()
    if arg == 'lu':
        m,n = A.numpy().shape
        if m == n:
            return linalg.solve(A.numpy(),b.numpy()), time.time() - t
        else:
            x, _,_,_ = linalg.lstsq(A.numpy(),b.numpy())
            return torch.from_numpy(x), time.time() - t
    elif arg == 'qr':

        A = A.numpy()
        b = b.numpy()
        [m, n] = A.shape




        b = b.reshape(m, 1)
        ## 1 Comupte QR via householder (Alg. 10.1 in T+B)
        ## AND
        ## 2 Compute Q^T b = y (Alg. 10.2 in T+B)


        ## building R and y (note to save memory A = R, b = y)

        for k in range(n):
            I = np.eye(m - k)
            x = np.copy(A[k:m, k].reshape(m - k, 1))
            v = np.sign(x[0]) * np.linalg.norm(x) * I[:, 0].reshape(m - k, 1) + x
            v = v / np.linalg.norm(v)
            A[k:m, k:n] = A[k:m, k:n] - 2 * np.dot(v, np.dot(np.transpose(v), A[k:m, k:n]))
            b[k:m] = b[k:m] - 2 * np.dot(v, np.dot(np.transpose(v), b[k:m]))

        ## solve Rx = y w/ back substitution (Alg. 17.1 in T+B)

        x = torch.zeros(n)
        x = x.numpy()

        for j in range(n - 1, -1, -1):
            sum = 0
            for k in range(j + 1, n, 1):
                sum += x[k] * A[j, k]
            x[j] = (b[j] - sum) / A[j, j]

        return torch.from_numpy(x), time.time() - t
    elif arg == 'normal':
        a = A.t() @ A
        y = A.t().mv(b)
        R = torch.from_numpy(linalg.cholesky(a.numpy()))
        w, _ = y.gels(R.t())
        x, _ = w.gels(R)
        x = x[:,0]
        return x, time.time() - t
    elif arg == 'svd':
        (u,s,v) = torch.svd(A)
        y = u.t() @ b
        w, _ = torch.gels(y,torch.diag(s))
        x = v @ w
        x = x[:,0]
        return x, time.time() - t
    else:
        print('Incorrect Arg')
        return 0, 0


def gmres(_A, _b, x0=None, max_iter=10, tol=1e-5):
    """
    solve a generalized linear system via GMRES
    TODO: find a least squares minimizer compatible w/ PyTorch
    
    :param x0: initial guess
    :param max_iter: number of iterations to run
    """
    A = _A.cpu()

    nrows = A.size(0)
    x = [torch.FloatTensor(nrows)] * max_iter
    Q = torch.FloatTensor(max_iter, nrows)
    if x0 is None:
        x0 = torch.zeros(nrows)

    Q[0] = _b / _b.norm()

    H = torch.zeros(max_iter+1, max_iter)

    for k in range(max_iter):
        y = A @ Q[k]
        for j in range(k):
            H[j, k] = Q[j].dot(y)
            y -= H[j, k] * Q[j]

        H[k+1, k] = y.norm()
        if (H[k+1, k] != 0) and (k != max_iter-1):
            Q[k+1] = y / H[k+1, k]

        b = torch.zeros(max_iter + 1)
        b[0] = _b.norm()

        result = np.linalg.lstsq(H.numpy(), b.numpy())
        result = torch.FloatTensor(result[0])

        x[k] = (Q.transpose(0, 1) @ result) + x0

    return x


def cg(A, b, x=None, max_iter=10, tol=1e-5, cuda=True):
    """
    solve a symmetric, positive definite linear system via conjugate
    gradient iteration
    
    :param x: initial guess
    :param max_iter: maximum allowable iterations
    :param tol: residual tolerance
    """
    m = A.size(0)
    if x is None:
        x = torch.zeros(m)
    if cuda:
        x = x.cuda()

    r = (A @ x) - b
    p = -r
    r_k_norm = r.dot(r)
    for i in range(max_iter):
        Ap = A @ p
        alpha = r_k_norm / p.dot(Ap)
        x += alpha * p
        r += alpha * Ap
        r_kplus1_norm = r.dot(r)
        beta = r_kplus1_norm / r_k_norm
        r_k_norm = r_kplus1_norm
        if r_kplus1_norm < tol:
            break
        p = beta * p - r

    return x
