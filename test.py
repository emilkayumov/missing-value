import numpy as np
import random_deletion
from data_recovery import svd_imputer


# test random deletion

X = np.ones((20, 10))

print(random_deletion.make_missing_value(X, 0.05, 0.2, 1.0))

svd_imputer(X)