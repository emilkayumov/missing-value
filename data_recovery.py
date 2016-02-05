import numpy as np
from scipy.stats.mstats import mode


def ignore_imputer(data, data_y=None, ignore_object=True):
    """
    A function for making the dataset without objects (or features) with mmissing values.
    ----------
    :param data: dataset
    :param data_y: target (optional)
    :param ignore_object: if true objects with missing values will be ignored, otherwise features will be ignored
    :return: X or X, y (if data_y will be)
    """
    if ignore_object:
        mask = np.sum(data != data, axis=1) == 0
        X = data[mask]
        if data_y:
            y = data_y[mask]
    else:
        mask = np.sum(data != data, axis=0) == 0
        X = data[:, mask]
        if data_y:
            y = data_y

    if data_y:
        return X, y
    else:
        return X


def special_value_imputer(data, value=-1):
    """
    A function for filling missing values in dataset with special value.
    :param data: dataset
    :param value: special value
    :return: dataset without missing values
    """
    X = np.array(data)
    mask = X != X
    X[mask] = value

    return X


def common_value_imputer(data):
    """
    A function for filling missing values in dataset with common value for each feature.
    :param data: dataset
    :return: dataset without missing values
    """
    X = np.array(data)
    for i in range(X.shape[1]):
        mask = X[:, i] != X[:, i]
        X[mask, i] = mode(X[np.logical_not(mask), i])[0][0]

    return X


def svd_imputer(data, rank=None, max_iter=30, tol=1e-1):
    # https://web.stanford.edu/~hastie/Papers/missing.pdf

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


def knn_imputer(data, n_neighbors=1, metric='l2'):
    X = np.array(data)
    mask = X != X
    objects = mask.sum(axis=1) != 0

    # without missing values
    X_full = X[np.logical_not(objects)]

    for i, obj in enumerate(objects):
        if not obj:
            continue

        mask_obj = np.logical_not(mask[i])

        # finding nearest
        if metric == 'l2':
            neighbors = ((X_full[:, mask_obj] - X[i, mask_obj]) ** 2).sum(axis=1).argsort()[:n_neighbors]
        elif metric == 'l1':
            neighbors = ((X_full[:, mask_obj] - X[i, mask_obj]).abs()).sum(axis=1).argsort()[:n_neighbors]
        else:
            distance = np.zeros(X_full.shape[0])
            for j in range(X_full.shape[0]):
                distance[j] = metric(X[i, mask_obj], X_full[j, mask_obj])
            neighbors = distance.argsort()[:n_neighbors]

        X_neighbors = X_full[neighbors]

        # replacing missing values by most common
        for j, feat in enumerate(mask_obj):
            if feat:
                continue

            X[i, j] = mode(X_neighbors[:, j])[0][0]

    return X
