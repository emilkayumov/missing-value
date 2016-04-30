import numpy as np


def make_missing_value(X, del_fraction=0.05, del_fraction_column=1.0, del_fraction_row=1.0, del_columns=None):
    """
    A function for making random missing value in dataset (MCAR).
    ----------
    :param X: dataset
    :param del_fraction: a fraction of missing value in all dataset
    :param del_fraction_column: a fraction of columns which has missing values
    :param del_fraction_row: a fraction of rows which has missing values
    :return: dataset with missing value
    """

    N = X.shape[0]
    D = X.shape[1]

    col_count = int(D * del_fraction_column)
    row_count = int(N * del_fraction_row)

    # choosing columns and rows
    del_columns = np.random.permutation(np.arange(D))[:col_count]
    
    if del_columns is None:
        del_columns = np.arange(D)[D-col_count:]
    del_row = np.random.permutation(np.arange(N))[:row_count]

    # calc new delete fraction as fraction of missing value in chosen columns and rows.
    new_del_fraction = del_fraction / (del_fraction_row * del_fraction_column)

    # new delete fraction = 1.0 means that all values from chosen columns and rows will be deleted.
    # if bigger than 1.0 change it to 0.5 and print warning with new global delete fraction.
    if new_del_fraction > 1.0:
        new_del_fraction = 0.5
        print('Warning: del_fraction is too big for del_fraction_column and del_fraction_row. ' +
              'It will be set to {0}.'.format(0.5 * del_fraction_column * del_fraction_row))

    # making mask for deletion
    delete_mask = np.array(np.random.random((N, D)) < new_del_fraction, dtype=int)
    delete_mask[del_row, :] += 1
    delete_mask[:, del_columns] += 1
    delete_mask = np.array(delete_mask == 3, dtype=bool)

    new_X = np.array(X)
    new_X[delete_mask] = np.nan

    return new_X
