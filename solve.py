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


"""
A = torch.randn(10,5)
xtrue = torch.randn(5)
b = A.mv(xtrue)

#test again scipy standard
x2, _,_,_ = linalg.lstsq(A.numpy(),b.numpy())
print(torch.norm(A @ torch.from_numpy(x2) -b))

x,t = solve(A,b,"QR")
print(torch.norm(A @ x - b))                #torch.from_numpy(x) - xtrue)/torch.norm(xtrue))
"""

