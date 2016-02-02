import numpy as np


def svd_imputer(data, rank=None, max_iter=30, tol=1e-1):

    # https://web.stanford.edu/~hastie/Papers/missing.pdf

    from scipy.stats.mstats import mode

    X = np.array(data)
    mask = X != X

    # first inputing by most common
    for i in range(X.shape[1]):
        X[mask[:, i], i] = mode(X[np.logical_not(mask[:, i]), i])[0][0]


    # iteratively using svd for best approximation
    for i in range(max_iter):

        U, s, V = np.linalg.svd(X, full_matrices=False)

        if rank:
            s[rank:] = 0

        new_X = U.dot(np.diag(s).dot(V))

        # check convergence
        if np.abs((new_X[mask] - X[mask]) / (X[mask] + 1e-10)).sum() / mask.sum() < tol:
            break
        X[mask] = new_X[mask]


    X[mask] = new_X[mask]

    return X
