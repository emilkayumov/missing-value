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


def common_value_imputer(data, add_binary=False):
    """
    A function for filling missing values in dataset with the most common value for each feature.
    :param data: dataset
    :param add_binary: adding additonal columns with mask missing or not
    :return: dataset without missing values
    """
    X = np.array(data)
    mask = X != X

    for col in range(X.shape[1]):
        X[mask[:, col], col] = mode(X[~mask[:, col], col])[0][0]

    if add_binary:
        X = _add_missing_binary(X, mask)

    return X


def mean_value_imputer(data, add_binary=False):
    """
    A function for filling missing values in dataset with mean value for each feature.
    :param data: dataset
    :param add_binary: adding additonal columns with mask missing or not
    :return: dataset without missing values
    """
    X = np.array(data)
    mask = X != X

    for col in range(X.shape[1]):
        X[mask[:, col], col] = np.mean(X[~mask[:, col], col])

    if add_binary:
        X = _add_missing_binary(X, mask)

    return X


def svd_imputer(data, rank=None, max_iter=10, tol=1e-1, round_nearest=True, add_binary=False):
    """
    A function for filling missing values in dataset with SVD.
    :param data: dataset
    :param rank: a rank of SVD
    :param max_iter: maximum number of iteration
    :param tol: tolerance of convergence
    :param round_nearest: rounding to the nearest value in array
    :param add_binary: adding additonal columns with mask missing or not
    :return: dataset without missing values
    """

    # https://web.stanford.edu/~hastie/Papers/missing.pdf

    X = np.array(data)
    mask = X != X

    # first inputing by most common/mean
    X = _first_imputer(X, mask)

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

    if round_nearest:
        X = _round_nearest(X, mask)

    if add_binary:
        X = _add_missing_binary(X, mask)

    return X


def knn_imputer(data, n_neighbors=1, metric='l2', round_nearest=True, add_binary=False):
    """
    A function for filling missing values in dataset with kNN.
    :param data: dataset
    :param n_neighbors: number of nearest neighbors for find most common/mean value
    :param metric: metric to find nearest neighbors (l2, l1 or own function)
    :param round_nearest: rounding to the nearest value in array
    :param add_binary: adding additonal columns with mask missing or not
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

        for j, feat in enumerate(mask_obj):
            if feat:
                continue

            if X_neighbors.shape[0] == 0:
                X[i, j] = np.mean(X[np.logical_not(mask[:, j]), j])
            else:
                X[i, j] = np.mean(X_neighbors[:, j])

    if round_nearest:
        X = _round_nearest(X, mask)

    if add_binary:
        X = _add_missing_binary(X, mask)

    return X


# base imputer for prediction methods
def predict_imputer(data, regressor, num_iters=3, verbosity=False, round_nearest=True, add_binary=False):
    """
    A function for filling missing values in dataset with Random Forest Regressor.
    :param data: dataset
    :param regressor: a class with fit, predict methods for imputing
    :param num_iters: a number of iteration for approximation
    :param verbosity: print information
    :param round_nearest: rounding to the nearest value in array
    :param add_binary: adding additonal columns with mask missing or not
    :return: dataset without missing values
    """

    X = data.copy()
    mask = X != X

    # first inputing by most common/mean
    X = _first_imputer(X, mask)

    # for exclusion of features
    feature_range = np.arange(X.shape[1])

    for it in range(num_iters):
        for i in range(X.shape[1]):

            if np.sum(mask[:, i]) > 0:
                regressor.fit(X[~mask[:, i], :][:, feature_range != i], X[~mask[:, i], i])
                X[mask[:, i], i] = regressor.predict(X[mask[:, i], :][:, feature_range != i])

            if verbosity:
                print('iter=' + str(it) + ' feat=' + str(i))

    if round_nearest:
        X = _round_nearest(X, mask)

    if add_binary:
        X = _add_missing_binary(X, mask)

    return X


def rf_imputer(data, num_iters=3, verbosity=False, round_nearest=True, add_binary=False):
    """
    A function for filling missing values in dataset with Random Forest Regressor.
    :param data: dataset
    :param num_iters: a number of iteration for approximation
    :param verbosity: print information
    :param round_nearest: rounding to the nearest value in array
    :param add_binary: adding additonal columns with mask missing or not
    :return: dataset without missing values
    """

    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators=10, n_jobs=-1)

    return predict_imputer(data, regressor, num_iters, verbosity, round_nearest, add_binary)


def linear_imputer(data, num_iters=3, verbosity=False, round_nearest=True, add_binary=False):
    """
    A function for filling missing values in dataset with Linear Regression.
    :param data: dataset
    :param num_iters: a number of iteration for approximation
    :param verbosity: print information
    :param round_nearest: rounding to the nearest value in array
    :param add_binary: adding additonal columns with mask missing or not
    :return: dataset without missing values
    """

    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression(normalize=True)

    return predict_imputer(data, regressor, num_iters, verbosity, round_nearest, add_binary)


def em_imputer(data, num_iters=3, verbosity=False, round_nearest=True, add_binary=False):
    """
    A function for filling missing values in dataset with EM.
    :param data: dataset
    :param num_iters: a number of iteration for approximation
    :param verbosity: print information
    :param round_nearest: rounding to the nearest value in array
    :param add_binary: adding additonal columns with mask missing or not
    :return: dataset without missing values
    """
    X = data.copy()
    mask = X != X

    X = _first_imputer(X, mask)

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

    if round_nearest:
        X = _round_nearest(X, mask)

    if add_binary:
        X = _add_missing_binary(X, mask)

    return X


def kmean_imputer(data, num_iters=3, verbosity=False, round_nearest=True, add_binary=False):
    """
    A function for filling missing values in dataset with K-Mean.
    :param data: dataset
    :param num_iters: a number of iteration for approximation
    :param verbosity: print information
    :param round_nearest: rounding to the nearest value in array
    :param add_binary: adding additonal columns with mask missing or not
    :return: dataset without missing values
    """
    X = data.copy()
    mask = X != X

    X = _first_imputer(X, mask)

    from sklearn.cluster import KMeans
    km = KMeans(n_jobs=1)

    for it in range(num_iters):

        km.fit(X)

        for row in range(X.shape[0]):
            if mask[row].sum():
                X[row, mask[row]] = km.cluster_centers_[km.labels_[row], mask[row]]

        if verbosity:
            print('iter', it + 1)

    if round_nearest:
        X = _round_nearest(X, mask)

    if add_binary:
        X = _add_missing_binary(X, mask)

    return X


def zet_imputer(data, competent_row_num, competent_col_num, round_nearest=True, add_binary=False):
    """
    A function for filling missing values in dataset with Zet algorithm.
    :param data: dataset (should be scaled)
    :param competent_row_num: number of competent rows
    :param competent_col_num: number of competent columns
    :param round_nearest: rounding to the nearest value in array
    :param add_binary: adding additonal columns with mask missing or not
    :return: dataset without missing values
    """

    # some bad things
    import warnings
    warnings.filterwarnings("ignore")

    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    X = data.copy()
    X_new = data.copy()
    mask = X != X

    for row in range(X.shape[0]):
        for col in range(X.shape[1]):
            if not mask[row, col]:
                continue
            row_missing = X[row, ~mask[row, :]].reshape((1, -1))
            col_missing = X[~mask[:, col], col].reshape((-1, 1))

            X_competent = X.copy()
            X_competent = X_competent[~mask[:, col], :]
            X_competent = X_competent[:, ~mask[row, :]]

            if row_missing.shape[0] == 0 or col_missing.shape[0] == 0:
                X_new[row, col] = 0
                continue

            weight1, weight2 = 0.5, 0.5
            b_row, b_col = 0, 0

            common_non_missing = np.logical_and(X_competent == X_competent, row_missing == row_missing)
            completeness = common_non_missing.sum(axis=1)
            distance = np.zeros(completeness.shape)
            for j in range(X_competent.shape[0]):
                distance[j] = ((X_competent[j, common_non_missing[j, :]] - row_missing[common_non_missing[j, :].reshape((1, -1))] ) ** 2).sum() ** 0.5
            competents_row = completeness / distance
            if X_competent.shape[0] > competent_row_num - 1:
                indexes = np.argsort(competents_row)[::-1][:competent_row_num - 1]
                X_competent = X_competent[indexes, :]
                col_missing = col_missing[indexes, :]
                competents_row = competents_row[indexes]

            common_non_missing = np.logical_and(X_competent == X_competent, col_missing == col_missing)
            completeness = common_non_missing.sum(axis=0)
            correlation = np.zeros(X_competent.shape[1])
            for j in range(X_competent.shape[1]):
                correlation[j] = np.corrcoef(X_competent[common_non_missing[:, j], j], col_missing[common_non_missing[:, j], 0])[0, 1]
                if np.isnan(correlation[j]):
                    correlation[j] = 0
            competents_col = completeness * np.abs(correlation)
            if X_competent.shape[1] > competent_col_num - 1:
                indexes = np.argsort(competents_col)[::-1][:competent_col_num - 1]
                X_competent = X_competent[:, indexes]
                row_missing = row_missing[:, indexes]
                competents_col = competents_col[indexes]

            X_competent[X_competent != X_competent] = 0
            alpha_range = np.arange(-3, 3, 1)

            if X_competent.shape[1] < 2:
                weight1 = 0
                weight2 = 1
            else:
                alpha_result = np.zeros(alpha_range.shape[0])
                for i, alpha in enumerate(alpha_range):
                    for c in range(X_competent.shape[1]):
                        bl = np.zeros(X_competent.shape[0])
                        mask_row = np.ones(X_competent.shape[1], dtype=bool)
                        mask_row[c] = False
                        x_train = X_competent[:, mask_row]
                        y_train = row_missing[:, mask_row].reshape(-1)
                        x_test = X_competent[:, c].reshape((-1, 1))
                        for r in range(X_competent.shape[0]):
                            lr.fit(x_train[r, :].reshape(1, -1).T, y_train)
                            bl[r] = lr.predict(x_test[r, :].reshape(1, -1))
                        b = (bl * competents_row ** alpha).sum() / (competents_row ** alpha).sum()
                        alpha_result[i] += np.abs(row_missing[:, c] - b)
                alpha_row = alpha_range[np.argmin(alpha_result)]

                bl = np.zeros(X_competent.shape[0])
                for r in range(X_competent.shape[0]):
                    lr.fit(X_competent[r, :].reshape((-1, 1)), row_missing.reshape(-1))
                    bl[r] = lr.predict(col_missing[r, :].reshape((-1, 1)))
                b_row = (bl * competents_row ** alpha_row).sum() / (competents_row ** alpha_row).sum()

            if X_competent.shape[0] < 2:
                if weight1 == 0:
                    X_new[row, col] = 0
                    continue
                else:
                    weight1 = 1
                    weight2 = 0
            else:
                alpha_result = np.zeros(alpha_range.shape[0])
                for i, alpha in enumerate(alpha_range):
                    for r in range(X_competent.shape[0]):
                        bl = np.zeros(X_competent.shape[1])
                        mask_col = np.ones(X_competent.shape[0], dtype=bool)
                        mask_col[r] = False
                        x_train = X_competent[mask_col, :]
                        y_train = col_missing[mask_col, :]
                        x_test = X_competent[r, :].reshape((1, -1))
                        for c in range(X_competent.shape[1]):
                            lr.fit(x_train[:, c].reshape(1, -1).T, y_train)
                            bl[c] = lr.predict(x_test[:, c].reshape(1, -1))
                        b = (bl * competents_col ** alpha).sum() / (competents_col ** alpha).sum()
                        alpha_result[i] += np.abs(col_missing[r, :] - b)
                alpha_col = alpha_range[np.argmin(alpha_result)]

                bl = np.zeros(X_competent.shape[1])
                for c in range(X_competent.shape[1]):
                    lr.fit(X_competent[:, c].reshape((-1, 1)), col_missing.reshape(-1))
                    bl[c] = lr.predict(row_missing[:, c].reshape((-1, 1)))
                b_col = (bl * competents_col ** alpha_col).sum() / (competents_col ** alpha_col).sum()

            X_new[row, col] = weight1 * b_row + weight2 * b_col

    if round_nearest:
        X_new = _round_nearest(X_new, mask)

    if add_binary:
        X_new = _add_missing_binary(X_new, mask)

    return X_new


# start with simple imputing with mean and find nearest
def _first_imputer(data, mask):

    for col in range(data.shape[1]):
        data[mask[:, col], col] = np.mean(data[~mask[:, col], col])

    data = _round_nearest(data, mask)

    return data


# find nearest in array
def _round_nearest(data, mask):

    for col in range(data.shape[1]):
        uniques = np.unique(data[~mask[:, col], col])
        for row in np.nonzero(mask[:, col])[0]:
            data[row, col] = _find_nearest(uniques, data[row, col])

    return data


# finding the nearest value in array
def _find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]


# add a binary column for every feature with missing or not
def _add_missing_binary(data, mask):

    # delete columns with no missing values
    add_mask = mask.copy()
    for col in range(mask.shape[1] - 1, -1, -1):
        if add_mask[:, col].sum() == 0:
            add_mask = np.delete(add_mask, col, axis=1)

    return np.hstack((data, np.array(add_mask, dtype=int)))