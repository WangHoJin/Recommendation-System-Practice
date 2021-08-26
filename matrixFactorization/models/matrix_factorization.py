import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy import sparse
from scipy.sparse import csr_matrix



def load(path):
    return pickle.load(open(path, "rb"))


def save(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


class MF():
    
    def __init__(self, output_path, verbose=False):
        # do nothing particularly
        self.verbose = verbose
        self.output_path = output_path
        pass
    
    def setData(self, R_train, R_valid, K, alpha, beta, num_iterations):
        self.R_train = R_train
        self.R_valid = R_valid
        self.num_users, self.num_movies = R_train.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.num_iterations = num_iterations

    def load_best(self):
        output_path = self.output_path
        self.U = np.loadtxt(output_path + '/U.dat')
        self.V = np.loadtxt(output_path + '/V.dat')
    

    def train(self):
        # Initialize user and item latent feature matrice
        self.U = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.V = np.random.normal(scale=1./self.K, size=(self.num_movies, self.K))
        self.T = []
        for i, j in zip(self.R_train.nonzero()[0], self.R_train.nonzero()[1]):
            self.T.append((i, j, self.R_train[i, j]))

        # Perform stochastic gradient descent for number of iterations
        endure_count = 5
        count = 0
        best_rmse = 9e7
        training_process = []
        for i in range(self.num_iterations):
            np.random.shuffle(self.T)
            self.sgd()
            rmse = self.eval_rmse()
            training_process.append((i, rmse))
            print("iteration: %d ; error = %.4f" % (i + 1, rmse))

            if rmse < best_rmse:
                np.savetxt(self.output_path + '/U.dat', self.U)
                np.savetxt(self.output_path + '/V.dat', self.V)
                best_rmse = rmse
                print("Best matrices are saved (err: {})".format(rmse))
            else:
                count = count + 1
            if count == endure_count:
                break
        return training_process

    def eval_rmse(self):
        xs, ys = self.R_valid.nonzero()
        predicted = self.U.dot(self.V.T)
        error = 0
        count = 0
        for x, y in zip(xs, ys):
            error += pow(self.R_valid[x, y] - predicted[y, x], 2)
            count = count + 1
        return np.sqrt(error)/count

    def sgd(self):
        for i, j, r in self.T:
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            U_i = self.U[i, :][:]

            self.U[i, :] += self.alpha * (2*e * self.V[j, :] - self.beta * self.U[i, :])
            self.V[j, :] += self.alpha * (2*e * U_i - self.beta * self.V[j, :])
    
    def get_rating(self, i, j):
        prediction = self.U[i, :].dot(self.V[j, :].T)
        return prediction



def train(res_dir, R_train, R_valid, max_iter=50, lambda_u=1, lambda_v=100, dimension=50, theta=None):
    model = MF(res_dir)
    model.setData(R_train, R_valid, K=dimension, alpha=0.01, beta=0.01, num_iterations=max_iter)
    training_process = model.train()
    model.load_best()
    R_predicted = model.U.dot(model.V.T) 
    return R_predicted


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_path", type=str, default='./data/tiny', help="Path input data pickle")
    parser.add_argument("-o", "--output_path", type=str, default='./', help="Output path")
    parser.add_argument("-m", "--max_iter", type=int, help="Max Iteration (default: 200)", default=200)
    parser.add_argument("-d", "--dim", type=int, help="Size of latent dimension (default: 50)", default=50)
    args = parser.parse_args()

    # seed setting
    np.random.seed(0)

    input_path = args.input_path
    if input_path is None:
        sys.exit("input_path is required")
    output_path = args.output_path
    if output_path is None:
        sys.exit("output_path is required")

    R_train = load(input_path + '/R_train.pkl')
    R_valid = load(input_path + '/R_valid.pkl')
    item_ids = load(input_path + '/item_ids.pkl')

    print("\nRun MF")

    model = MF(args.output_path)
    model.setData(R_train, R_valid, K=args.dim, alpha=0.01, beta=0.01, num_iterations=args.max_iter)
    training_process = model.train()

    model.load_best()
    R_predicted = model.self.U.dot(self.V.T)
    
    print("U x V:")
    print(R_predicted)
    print("Best valid error = %.4f" % (model.eval_rmse()))
