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
        return training_process

    def eval_rmse(self):
        return rmse

    def sgd(self):
        pass
    
    def get_rating(self, i, j):
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
