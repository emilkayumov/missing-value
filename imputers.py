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
        if data_y is not None:
            y = data_y[mask]
    else:
        mask = np.sum(data != data, axis=0) == 0
        X = data[:, mask]
        if data_y:
            y = data_y

    if data_y is not None:
        return X, y
    else:
        return X


def special_value_imputer(data, value=-1, add_binary=False):
    """
    A function for filling missing values in dataset with special value.
    :param data: dataset
    :param value: special value
    :return: dataset without missing values
    """
    X = np.array(data)
    mask = X != X
    X[mask] = value

    if add_binary:
        X = _add_missing_binary(X, mask)

    return X


def common_value_imputer(data, categorical_mask, add_binary=False):
    """
    A function for filling missing values in dataset with common/mean value for each feature.
    :param data: dataset
    :param categorical_mask: mask of columns in datasets - if true it is categorical feature
    :return: dataset without missing values
    """
    X = np.array(data)
    mask = X != X

    X = _first_imputer(X, categorical_mask, mask)

    if add_binary:
        X = _add_missing_binary(X, mask)

    return X


def svd_imputer(data, categorical_mask, rank=None, max_iter=30, tol=1e-1, makeround=False, add_binary=False):
    """
    A function for filling missing values in dataset with SVD.
    :param data: dataset
    :param categorical_mask: mask of columns in datasets - if true it is categorical feature
    :param rank: a rank of SVD
    :param max_iter: maximum number of iteration
    :param tol: tolerance of convergence
    :return: dataset without missing values
    """

    # https://web.stanford.edu/~hastie/Papers/missing.pdf

    X = np.array(data)
    mask = X != X

    # first inputing by most common/mean
    X = _first_imputer(X, categorical_mask, mask)

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

    X = _finish_imputer(X, categorical_mask, mask, makeround)

    if add_binary:
        X = _add_missing_binary(X, mask)

    return X


def knn_imputer(data, categorical_mask, n_neighbors=1, metric='l2', makeround=False, add_binary=False):
    """
    A function for filling missing values in dataset with kNN.
    :param data: dataset
    :param categorical_mask: mask of columns in datasets - if true it is categorical feature
    :param n_neighbors: number of nearest neighbors for find most common/mean value
    :param metric: metric to find nearest neighbors (l2, l1 or own function)
    :return: dataset without missing values
    """
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

            if categorical_mask[j]:
                if X_neighbors.shape[0] == 0:
                    # if no neighbors with no missing
                    X[i, j] = mode(X[np.logical_not(mask[:, j]), j])[0][0]
                else:
                    X[i, j] = mode(X_neighbors[:, j])[0][0]
            else:
                if X_neighbors.shape[0] == 0:
                    X[i, j] = np.mean(X[np.logical_not(mask[:, j]), j])
                else:
                    X[i, j] = np.mean(X_neighbors[:, j])

    if add_binary:
        X = _add_missing_binary(X, mask)

    return X


# base imputer for prediction methods
def predict_imputer(data, categorical_mask, regressor, num_iters=3, verbosity=False, makeround=False, add_binary=False):
    """
    A function for filling missing values in dataset with Random Forest Regressor.
    :param data: dataset
    :param categorical_mask: mask of columns in datasets - if true it is categorical feature
    :param regressor: a class with fit, predict methods for imputing
    :param num_iters: a number of iteration for approximation
    :param verbosity: print information
    :param makeround: filling only with values from dataset
    :return: dataset without missing values
    """

    X = data.copy()
    mask = X != X

    # first inputing by most common/mean
    X = _first_imputer(X, categorical_mask, mask)

    # for exclusion of features
    feature_range = np.arange(X.shape[1])

    for it in range(num_iters):
        for i in range(X.shape[1]):

            if np.sum(mask[:, i]) > 0:
                regressor.fit(X[~mask[:, i], :][:, feature_range != i], X[~mask[:, i], i])
                X[mask[:, i], i] = regressor.predict(X[mask[:, i], :][:, feature_range != i])

            if verbosity:
                print('iter=' + str(it) + ' feat=' + str(i))

    X = _finish_imputer(X, categorical_mask, mask, makeround)

    if add_binary:
        X = _add_missing_binary(X, mask)

    return X


def rf_imputer(data, categorical_mask, num_iters=3, verbosity=False, makeround=False, add_binary=False):
    """
    A function for filling missing values in dataset with Random Forest Regressor.
    :param data: dataset
    :param categorical_mask: mask of columns in datasets - if true it is categorical feature
    :param num_iters: a number of iteration for approximation
    :param verbosity: print information
    :param makeround: filling only with values from dataset
    :return: dataset without missing values
    """

    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators=10, n_jobs=-1)

    return predict_imputer(data, categorical_mask, regressor, num_iters, verbosity, makeround, add_binary)


def linear_imputer(data, categorical_mask, num_iters=3, verbosity=False, makeround=False, add_binary=False):
    """
    A function for filling missing values in dataset with Linear Regression.
    :param data: dataset
    :param categorical_mask: mask of columns in datasets - if true it is categorical feature
    :param num_iters: a number of iteration for approximation
    :param verbosity: print information
    :param makeround: filling only with values from dataset
    :return: dataset without missing values
    """

    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression(normalize=True)

    return predict_imputer(data, categorical_mask, regressor, num_iters, verbosity, makeround, add_binary)


def em_imputer(data, categorical_mask, num_iters=3, verbosity=False, makeround=False, add_binary=False):
    """
    A function for filling missing values in dataset with EM.
    :param data: dataset
    :param categorical_mask: mask of columns in datasets - if true it is categorical feature
    :param num_iters: a number of iteration for approximation
    :param verbosity: print information
    :param makeround: filling only with values from dataset
    :return: dataset without missing values
    """
    X = data.copy()
    mask = X != X

    X = _first_imputer(X, categorical_mask, mask)

    from sklearn.mixture import GMM
    gmm = GMM(covariance_type='full')

    for it in range(num_iters):

        gmm.fit(X)

        for row in range(X.shape[0]):
            if mask[row].sum():

                inv_cov = np.linalg.inv(gmm.covars_[0, ~ mask[row]][:, ~ mask[row]])
                delta = X[row, ~ mask[row]] - gmm.means_[0, ~ mask[row]]
                coef = gmm.covars_[0, mask[row]][:, ~ mask[row]].dot(inv_cov)

                X[row, mask[row]] = gmm.means_[0, mask[row]] + coef.dot(delta)

        if verbosity:
            print('iter', it + 1)


    X = _finish_imputer(X, categorical_mask, mask, makeround)

    if add_binary:
        X = _add_missing_binary(X, mask)

    return X


def kmean_imputer(data, categorical_mask, num_iters=3, verbosity=False, makeround=False, add_binary=False):
    """
    A function for filling missing values in dataset with K-Mean.
    :param data: dataset
    :param categorical_mask: mask of columns in datasets - if true it is categorical feature
    :param num_iters: a number of iteration for approximation
    :param verbosity: print information
    :param makeround: filling only with values from dataset
    :return: dataset without missing values
    """
    X = data.copy()
    mask = X != X

    X = _first_imputer(X, categorical_mask, mask)

    from sklearn.cluster import KMeans
    km = KMeans(n_jobs=1)

    for it in range(num_iters):

        km.fit(X)

        for row in range(X.shape[0]):
            if mask[row].sum():
                X[row, mask[row]] = km.cluster_centers_[km.labels_[row], mask[row]]

        if verbosity:
            print('iter', it + 1)


    X = _finish_imputer(X, categorical_mask, mask, makeround)

    if add_binary:
        X = _add_missing_binary(X, mask)

    return X


# start with simple imputing
def _first_imputer(data, categorical_mask, mask):

    for i in range(data.shape[1]):
        if categorical_mask[i]:
            data[mask[:, i], i] = mode(data[np.logical_not(mask[:, i]), i])[0][0]
        else:
            data[mask[:, i], i] = np.mean(data[np.logical_not(mask[:, i]), i])

    return data


# round if categorical or find nearest in array
def _finish_imputer(data, categorical_mask, mask, makeround):

    if makeround:
        for i in range(data.shape[1]):
            uniques = np.unique(data[~mask[:, i], i])
            for j in np.nonzero(mask[:, i])[0]:
                data[j, i] = _find_nearest(uniques, data[j, i])
    else:
        for i in range(data.shape[1]):
            if categorical_mask[i]:
                data[mask[:, i], i] = np.round(data[mask[:, i], i])

    return data


# finding the nearest value in array
def _find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]


# add a binary column for every feature with missing or not
def _add_missing_binary(data, mask):
    return np.hstack((data, np.array(mask, dtype=int)))