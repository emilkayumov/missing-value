import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import imputers
import random_deletion

from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


_rf = RandomForestClassifier(n_jobs=-1, criterion='entropy', random_state=123)
_lr = LogisticRegression(random_state=123)
_nn = KNeighborsClassifier(n_jobs=-1)

_algs = ['RF', 'LR', 'kNN']
_methods = ['ignore', 'special', 'common', 'mean', 'svd', 'knn', 'rf', 'lr', 'em', 'k-means', 'zet']
_colors = {'zet': '#CC0000',
           'special': '#888800',
           'common': '#66CC00',
           'mean': '#00CCCC',
           'svd': '#0066CC',
           'knn': '#6600CC',
           'rf': '#CC00CC',
           'lr': '#333333',
           'em': '#FF8000',
           'k-means': '#CCCC00'}
_datasets = ['krkp', 'creditg', 'segment']


def multi_algs_cv(data, y, cv):
    rf_cv = np.mean(cross_val_score(_rf, data, y, scoring='accuracy', cv=cv))
    lr_cv = np.mean(cross_val_score(_lr, data, y, scoring='accuracy', cv=cv))
    nn_cv = np.mean(cross_val_score(_nn, data, y, scoring='accuracy', cv=cv))
    return rf_cv, lr_cv, nn_cv


def dataset_exps(data, y, cv, add_binary):

    data_ignore, y_ignore = imputers.ignore_imputer(data, y)
    data_special_1 = imputers.special_value_imputer(data, -1, add_binary=add_binary)
    data_special_2 = imputers.special_value_imputer(data, 0, add_binary=add_binary)
    data_common = imputers.common_value_imputer(data, add_binary=add_binary)
    data_mean = imputers.mean_value_imputer(data, add_binary=add_binary)
    data_svd = imputers.svd_imputer(data, rank=data.shape[1] // 2, add_binary=add_binary)
    data_knn = imputers.knn_imputer(data, n_neighbors=5, add_binary=add_binary)
    data_rf = imputers.rf_imputer(data, add_binary=add_binary)
    data_lr = imputers.linear_imputer(data, add_binary=add_binary)
    data_em = imputers.em_imputer(data, add_binary=add_binary)
    data_km = imputers.kmean_imputer(data, add_binary=add_binary)
    data_zet = imputers.zet_imputer(data, competent_row_num=6, competent_col_num=4, add_binary=add_binary)

    result = np.zeros((len(_methods), len(_algs)))

    if data_ignore.shape[0] >= 10:
        result[0] = multi_algs_cv(data_ignore, y_ignore, 10)

    result[1, 0] = multi_algs_cv(data_special_1, y, cv)[0]
    result[1, 1:] = multi_algs_cv(data_special_2, y, cv)[1:]
    result[2] = multi_algs_cv(data_common, y, cv)
    result[3] = multi_algs_cv(data_mean, y, cv)
    result[4] = multi_algs_cv(data_svd, y, cv)
    result[5] = multi_algs_cv(data_knn, y, cv)
    result[6] = multi_algs_cv(data_rf, y, cv)
    result[7] = multi_algs_cv(data_lr, y, cv)
    result[8] = multi_algs_cv(data_em, y, cv)
    result[9] = multi_algs_cv(data_km, y, cv)
    result[10] = multi_algs_cv(data_zet, y, cv)

    result = pd.DataFrame(result, columns=_algs, index=_methods)

    if data_ignore.shape[0] < 10:
        result.drop('ignore', inplace=True)

    return result


def make_experiments(data_real, target, clf, cv, missing_frac_range, num_iter, sp_value, add_binary, del_columns=None):

    accuracy = pd.DataFrame(np.zeros((len(_methods), len(missing_frac_range))), index=_methods, columns=missing_frac_range)
    rmse = pd.DataFrame(np.zeros((len(_methods), len(missing_frac_range))), index=_methods, columns=missing_frac_range)

    for missing_frac in missing_frac_range:
        print('start fraction:', missing_frac)
        
        for iteration in range(num_iter):
            data_missing = random_deletion.make_missing_value(data_real, del_fraction=missing_frac,
                                                              del_fraction_column=0.5)

            # ignore
            data_imp, y = imputers.ignore_imputer(data_missing, target)
            if data_imp.shape[0] >= data_missing.shape[0] / 10:
                cur_accuracy = np.mean(cross_val_score(clf, data_imp, y, scoring='accuracy', cv=10))
            else:
                cur_accuracy = 0

            if iteration == 0:
                accuracy.ix['ignore', missing_frac] = cur_accuracy / num_iter
            elif cur_accuracy == 0 or accuracy.ix['ignore', missing_frac] == 0:
                accuracy.ix['ignore', missing_frac] = 0
            else:
                accuracy.ix['ignore', missing_frac] += cur_accuracy / num_iter

            # special value
            data_imp = imputers.special_value_imputer(data_missing, value=sp_value, add_binary=add_binary)
            cur_accuracy = np.mean(cross_val_score(clf, data_imp, target, scoring='accuracy', cv=cv))
            cur_rmse = np.sum(np.array((data_real - data_imp) ** 2)) ** 0.5
            accuracy.ix['special', missing_frac] += cur_accuracy / num_iter
            rmse.ix['special', missing_frac] += cur_rmse / num_iter

            # common value
            data_imp = imputers.common_value_imputer(data_missing, add_binary=add_binary)
            cur_accuracy = np.mean(cross_val_score(clf, data_imp, target, scoring='accuracy', cv=cv))
            cur_rmse = np.sum(np.array((data_real - data_imp) ** 2)) ** 0.5
            accuracy.ix['common', missing_frac] += cur_accuracy / num_iter
            rmse.ix['common', missing_frac] += cur_rmse / num_iter

            # mean value
            data_imp = imputers.mean_value_imputer(data_missing, add_binary=add_binary)
            cur_accuracy = np.mean(cross_val_score(clf, data_imp, target, scoring='accuracy', cv=cv))
            cur_rmse = np.sum(np.array((data_real - data_imp) ** 2)) ** 0.5
            accuracy.ix['mean', missing_frac] += cur_accuracy / num_iter
            rmse.ix['mean', missing_frac] += cur_rmse / num_iter

            # svd
            data_imp = imputers.svd_imputer(data_missing, rank=data_missing.shape[1] // 2, add_binary=add_binary)
            cur_accuracy = np.mean(cross_val_score(clf, data_imp, target, scoring='accuracy', cv=cv))
            cur_rmse = np.sum(np.array((data_real - data_imp) ** 2)) ** 0.5
            accuracy.ix['svd', missing_frac] += cur_accuracy / num_iter
            rmse.ix['svd', missing_frac] += cur_rmse / num_iter

            # knn
            data_imp = imputers.knn_imputer(data_missing, n_neighbors=5, add_binary=add_binary)
            cur_accuracy = np.mean(cross_val_score(clf, data_imp, target, scoring='accuracy', cv=cv))
            cur_rmse = np.sum(np.array((data_real - data_imp) ** 2)) ** 0.5
            accuracy.ix['knn', missing_frac] += cur_accuracy / num_iter
            rmse.ix['knn', missing_frac] += cur_rmse / num_iter

            # rf
            data_imp = imputers.rf_imputer(data_missing, add_binary=add_binary)
            cur_accuracy = np.mean(cross_val_score(clf, data_imp, target, scoring='accuracy', cv=cv))
            cur_rmse = np.sum(np.array((data_real - data_imp) ** 2)) ** 0.5
            accuracy.ix['rf', missing_frac] += cur_accuracy / num_iter
            rmse.ix['rf', missing_frac] += cur_rmse / num_iter

            # lr
            data_imp = imputers.linear_imputer(data_missing, add_binary=add_binary)
            cur_accuracy = np.mean(cross_val_score(clf, data_imp, target, scoring='accuracy', cv=cv))
            cur_rmse = np.sum(np.array((data_real - data_imp) ** 2)) ** 0.5
            accuracy.ix['lr', missing_frac] += cur_accuracy / num_iter
            rmse.ix['lr', missing_frac] += cur_rmse / num_iter

            # em
            data_imp = imputers.em_imputer(data_missing, add_binary=add_binary)
            cur_accuracy = np.mean(cross_val_score(clf, data_imp, target, scoring='accuracy', cv=cv))
            cur_rmse = np.sum(np.array((data_real - data_imp) ** 2)) ** 0.5
            accuracy.ix['em', missing_frac] += cur_accuracy / num_iter
            rmse.ix['em', missing_frac] += cur_rmse / num_iter

            # km
            data_imp = imputers.kmean_imputer(data_missing, add_binary=add_binary)
            cur_accuracy = np.mean(cross_val_score(clf, data_imp, target, scoring='accuracy', cv=cv))
            cur_rmse = np.sum(np.array((data_real - data_imp) ** 2)) ** 0.5
            accuracy.ix['k-means', missing_frac] += cur_accuracy / num_iter
            rmse.ix['k-means', missing_frac] += cur_rmse / num_iter

            # zet
            data_imp = imputers.zet_imputer(data_missing, competent_row_num=6, competent_col_num=4, add_binary=add_binary)
            cur_accuracy = np.mean(cross_val_score(clf, data_imp, target, scoring='accuracy', cv=cv))
            cur_rmse = np.sum(np.array((data_real - data_imp) ** 2)) ** 0.5
            accuracy.ix['zet', missing_frac] += cur_accuracy / num_iter
            rmse.ix['zet', missing_frac] += cur_rmse / num_iter

    return accuracy, rmse


def make_plots_accuracy(accuracy_rf, accuracy_lr, accuracy_knn, dataset_name, filename=''):
    plt.figure(figsize=(20, 5))
    plt.suptitle('Accuracy (' + dataset_name + ')', fontsize=18)

    plt.subplot(1, 4, 1)
    for method in accuracy_rf.index:
        plt.plot(accuracy_rf.columns, accuracy_rf.ix[method], label=method, color=_colors[method], lw=1.5)
    plt.title('Random forest', fontsize=14)
    plt.xlabel("Missing value fraction", fontsize=12)
    plt.xlim([0, 0.15])
    plt.ylabel("Accuracy", fontsize=12)

    plt.subplot(1, 4, 2)
    for method in accuracy_lr.index:
        plt.plot(accuracy_lr.columns, accuracy_lr.ix[method], label=method, color=_colors[method], lw=1.5)
    plt.title('Logistic regression', fontsize=14)
    plt.xlabel("Missing value fraction", fontsize=12)
    plt.xlim([0, 0.15])
    #plt.ylabel("Accuracy", fontsize=12)

    plt.subplot(1, 4, 3)
    for method in accuracy_knn.index:
        plt.plot(accuracy_knn.columns, accuracy_knn.ix[method], label=method, color=_colors[method], lw=1.5)
    plt.title('Nearest neighbors', fontsize=14)
    plt.xlabel("Missing value fraction", fontsize=12)
    plt.xlim([0, 0.15])
    #plt.ylabel("Accuracy", fontsize=12)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
    if filename:
        plt.savefig(filename)
    plt.show()


def make_plots_rmse(rmse_krkp, rmse_creditg, rmse_segment, filename):
    plt.figure(figsize=(20, 5))
    plt.suptitle('RMSE', fontsize=18)

    plt.subplot(1, 4, 1)
    for method in rmse_krkp.index:
        plt.plot(rmse_krkp.columns, rmse_krkp.ix[method], label=method, color=_colors[method], lw=1.5)
    plt.title(_datasets[0], fontsize=14)
    plt.xlim([0, 0.15])
    plt.xlabel("Missing value fraction", fontsize=12)
    plt.ylabel("RMSE", fontsize=12)

    plt.subplot(1, 4, 2)
    for method in rmse_creditg.index:
        plt.plot(rmse_creditg.columns, rmse_creditg.ix[method], label=method, color=_colors[method], lw=1.5)
    plt.title(_datasets[1], fontsize=14)
    plt.xlim([0, 0.15])
    plt.xlabel("Missing value fraction", fontsize=12)
    #plt.ylabel("RMSE", fontsize=12)

    plt.subplot(1, 4, 3)
    for method in rmse_segment.index:
        plt.plot(rmse_segment.columns, rmse_segment.ix[method], label=method, color=_colors[method], lw=1.5)
    plt.title(_datasets[2], fontsize=14)
    plt.xlim([0, 0.15])
    plt.xlabel("Missing value fraction", fontsize=12)
    #plt.ylabel("RMSE", fontsize=12)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
    if filename:
        plt.savefig(filename)
    plt.show()