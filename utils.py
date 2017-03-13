
import numpy as np


# vectorize a matrix
def vec(matrices, axis=1):
    return np.concatenate([a.T.flatten() if axis else a.flatten()
                           for a in matrices])

# recover original model parameter structure
def format_params(params_vec, dims):
    from functools import reduce
    from operator import mul

    params = []
    j = 0
    for dim in dims:
        n = reduce(mul, dim)
        param = params_vec[j:(j+n)]
        if len(dim) > 1:
            param = param.reshape(dim[1], dim[0])
        params.append(param.T)
        j += n
    return params