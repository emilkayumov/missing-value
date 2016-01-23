import numpy as np
import random_deletion

# test random deletion

X = np.ones((20, 10))

print(random_deletion.make_missing_value(X, 0.05, 0.2, 1.0))
