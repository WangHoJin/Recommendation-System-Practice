import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


def compute_sim(R):
    num_users = R.shape[0]
    UbyU = (R * R.transpose()).toarray()
    UbyU[range[num_users], range(num_users)] = 0
    return UbyU

def compute_sim_cosine(R):
    num_users = R.shape[0]
    ones = csr_matrix((np.ones(R.nnz), (R.nonzero()[0], R.nonzero()[1])), shape=R.shape)
    
    UbyU = (R * R.transpose()).toarray()
    UbyU[range(num_users), range(num_users)] = 0

    sim = np.zeros((num_users, num_users))
    for i in range(num_users):
        both = ones[i].reshape(1, -1).multiply(ones)
        norm_i = norm(both.multiply(R[i].reshape(1, -1)), ord=2, axis=1)
        norm_others = norm(both.multiply(R), ord=2, axis=1)
        den = norm_i * norm_others
        den[den == 0] = 1
        sim[i] = UbyU[i] / den
    return sim

def predict(R, K):
    num_users = R.shape[0]
    num_items = R.shape[1]

    sim = compute_sim(R)
    topk = sim.argsort(axis = 1)[:, -K:]
    R_predicted = np.zeros((num_users, num_items))
    for i in range(num_users):
        weights = sim[i, topk[i]]
        for j in range(K):
            R_predicted[i] += weights[j] * R[topk[i, j]]
        R_predicted[i] /= weights.sum()
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
