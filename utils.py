import numpy as np
import pandas as pd
import imputers
import random_deletion

from sklearn.cross_validation import cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

_rf = RandomForestClassifier(n_jobs=-1, random_state=123)
_lr = LogisticRegression(random_state=123)
_nn = KNeighborsClassifier(n_jobs=-1)

_algs = ['RF', 'LR', 'kNN']
_methods = ['ignore', 'special', 'common', 'svd', 'knn', 'rf', 'lr', 'em', 'k-means']


def multi_algs_cv(data, y, cv):
    rf_cv = np.mean(cross_val_score(_rf, data, y, scoring='accuracy', cv=cv))
    lr_cv = np.mean(cross_val_score(_lr, data, y, scoring='accuracy', cv=cv))
    nn_cv = np.mean(cross_val_score(_nn, data, y, scoring='accuracy', cv=cv))
    return rf_cv, lr_cv, nn_cv


def dataset_exps(data, y, cv):

    data_ignore, y_ignore = imputers.ignore_imputer(data, y)
    data_special = imputers.special_value_imputer(data, -1)
    data_common = imputers.common_value_imputer(data)
    data_svd = imputers.svd_imputer(data, rank=data.shape[1] // 2, max_iter=3)
    data_knn = imputers.knn_imputer(data, n_neighbors=5)
    data_rf = imputers.rf_imputer(data)
    data_lr = imputers.linear_imputer(data)
    data_em = imputers.em_imputer(data)
    data_km = imputers.kmean_imputer(data)

    result = np.zeros((len(_methods), len(_algs)))

    if data_ignore.shape[0] >= 10:
        result[0] = multi_algs_cv(data_ignore, y_ignore, 10)

    result[1] = multi_algs_cv(data_special, y, cv)
    result[2] = multi_algs_cv(data_common, y, cv)
    result[3] = multi_algs_cv(data_svd, y, cv)
    result[4] = multi_algs_cv(data_knn, y, cv)
    result[5] = multi_algs_cv(data_rf, y, cv)
    result[6] = multi_algs_cv(data_lr, y, cv)
    result[7] = multi_algs_cv(data_em, y, cv)
    result[8] = multi_algs_cv(data_km, y, cv)

    result = pd.DataFrame(result, columns=_algs, index=_methods)

    if data_ignore.shape[0] < 10:
        result.drop('ignore', inplace=True)

    return result




# def make_expirement_method(estimator, method, param, list_data_missing, data_real, y, scorer, cv):
#
#     scores = np.zeros(len(list_data_missing))
#     rmse   = np.zeros(len(list_data_missing))
#
#     for data_missing in list_data_missing:
#
#         data_filled = method(data_missing) # fix
#
#         scores.append(np.mean(cross_val_score(estimator, data_filled, y, scoring=scorer, cv=cv, n_jobs=-1)))
#         rmse.append(np.sum((data_filled - data_real) ** 2) ** 0.5)
#
#     return scores, rmse
#
#
# def make_list_missing(data_real, del_fraction_range, del_fraction_column=1.0, del_fraction_row=1.0):
#
#     list_data_missing = []
#
#     for del_fraction in del_fraction_range:
#         list_data_missing.append(random_deletion.make_missing_value(data_real, del_fraction, del_fraction_column,
#                                                                     del_fraction_row))
#
#     return list_data_missing
#
#
# def make_expirement_all(estimator, list_method, list_param):
#     pass