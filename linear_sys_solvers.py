import numpy as np
import time


def qr(A, b):
    """
    :param A: m by n matrix that is that 'data'
    :param b: n vector that is the 'output'
    :return: m vector that is the 'parameters'
    """
    A = np.asarray(A, dtype = np.float32)
    b = np.asarray(b, dtype = np.float32)

    [m, n] = A.shape
    b = b.reshape(m,1)
    ## 1 Comupte QR via householder (Alg. 10.1 in T+B)
    ## AND
    ## 2 Compute Q^T b = y (Alg. 10.2 in T+B)


    ## building R and y (note to save memory A =R, b = y)

    for k in range(n):
        I = np.eye(m - k)
        x = np.copy(A[k:m, k].reshape(m-k,1))
        v = np.sign(x[0]) * np.linalg.norm(x) * I[:, 0].reshape(m - k, 1) + x
        v = v / np.linalg.norm(v)
        A[k:m, k:n] = A[k:m, k:n] - 2 * np.dot(v, np.dot(np.transpose(v), A[k:m, k:n]))
        b[k:m] = b[k:m] - 2 * np.dot(v, np.dot(np.transpose(v), b[k:m]))

    ## solve Rx = y w/ back substitution (Alg. 17.1 in T+B)
    x = np.zeros(n)
    sum = 0
    for j in range(n-1, 0, -1):
        sum = 0
        for k in range(j + 1, n, 1):
            sum += x[k] * A[j, k]
        x[j] = (b[j] - sum) / A[j, j]
    return x

## FIX ME
def svd(A, b):
    """
        :param A: m by n matrix that is that 'data'
        :param b: n vector that is the 'output'
        :return: m vector that is the 'parameters'
        """
    # 1 compute the SVD

    # 2 compute the U_hat * b

    # 3 compute
    x = 0
    return x

## FIX ME
def lu(A, b):
    """
        :param A: m by n matrix that is that 'data'
        :param b: n vector that is the 'output'
        :return: m vector that is the 'parameters'
        """
    x = 0
    return x


A = np.array([[3,1],[1,2]])
b = np.array([1,2])

t = time.time()
x1 = np.linalg.solve(A,b)
elapsed = time.time() - t
print(x1,elapsed)

t = time.time()
x2 = qr(A,b)
elapsed = time.time() - t

print(x2,elapsed)