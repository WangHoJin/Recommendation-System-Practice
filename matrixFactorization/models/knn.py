import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


def compute_sim(R):
    return UbyU


def predict(R, K):
    return R_predicted

if __name__ == '__main__':
    R = csr_matrix(np.array([
        [1, 2, 0],
        [1, 0, 3],
        [3, 2, 1],
        [3, 2, 2],
        [1, 2, 2],
    ]))
    R_predicted = predict(R, 2)
