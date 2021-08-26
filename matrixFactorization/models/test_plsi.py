import argparse
import sys
import pickle
import numpy as np
import time


class pLSI_dense:
    # implementation of probablistic Latent Semantic Indexing

    def __init__(self, verbose=False):
        # do nothing particularly
        self.verbose = verbose
        pass

    def setData(self, dw_matrix):
        # dw_matrix is required to be given in a two dimensional numpy array (nDocuments,nVocabulary)
        # with each element representing the number of times observed

        # set parameters
        self.dw_matrix = dw_matrix
        self.nDocuments = dw_matrix.shape[0]
        self.nVocabulary = dw_matrix.shape[1]
    
    def compute_observed(self):
        return observed

    def update_pZD(self, observed):
        pass             

    def update_pWZ(self, observed):
        pass
 
    def update_pZDW(self):
        pass

    def compute_log_likelihood_slowest(self):
        return log_likelihood
    

    def compute_log_likelihood_fast(self):
        return log_likelihood


    def solve(self, nTopics=10, max_iter=5, epsilon=1e-6):

        # set additional parameters
        self.nTopics = nTopics
        self.epsilon = epsilon

        # Randomly initialize
        self.pZD = np.random.rand(self.nTopics, self.nDocuments)
        self.pZD /= self.pZD.sum(axis=0).reshape(1, -1)
        
        self.pWZ = np.random.rand(self.nVocabulary, self.nTopics)
        self.pWZ /= self.pWZ.sum(axis=0).reshape(1, -1)

        self.pZDW = np.random.rand(self.nTopics, self.nDocuments, self.nVocabulary)
        self.pZDW /= self.pZDW.sum(axis=0).reshape((1, self.nDocuments, self.nVocabulary))


        #assert np.sum(self.pZDW[:, 0, 0]) == 1., "check probability sum"


        # start solving using EM algorithm
        for iter in range(max_iter):
            tic = time.time()
            delta = 0.0

            ### M step ###
            # common terms
            observed = self.compute_observed()

            # update pWZ
            self.update_pWZ(observed)
            
            # update pZD
            self.update_pZD(observed)

            toc = time.time()
            print(f"M-step took {toc-tic:.1f} seconds")
            
            ### E step ###
            self.update_pZDW()

            tac = time.time()
            print(f"E-step took {tac-toc:.1f} seconds")
            
            print("Compute log-likelihood")
            #log_likelihood = self.compute_log_likelihood_slow()
            log_likelihood = self.compute_log_likelihood_fast()
            print("log-likelihood", log_likelihood)
            
            ### break if converged ###
            if iter > 0:
                delta = log_likelihood - previousLL
                if delta < 0.:
                    print("likelihood decreased. Thus, Stop.")
                    break
                elif abs(delta) / abs(log_likelihood) < epsilon:
                    print("Log-likelihood changes under %f %% in this iteration. Thus, Stop." % (epsilon * 100))
                    break
                print(f"Iteration @ {iter} : Prev Log-likelihood {previousLL:.1f} -> Log-likelihood {log_likelihood:.1f}. Took {time.time()-tic:.1f} seconds")
            previousLL = log_likelihood
            
        return self.pZD.T

def gen_document_word_frequency(input_path):
    return dw_matrix

def train_dense_pLSI(input_path, nTopics):
    dw_matrix = gen_document_word_frequency(input_path)

    # apply probablistic semantic indexing
    model = pLSI_dense()
    model.setData(dw_matrix)

    dz_matrix = model.solve(nTopics=nTopics, max_iter=5, epsilon=1e-5)
    return dz_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_path", type=str, required=True, help="Path input data pickle")
    parser.add_argument("-o", "--output_path", type=str, required=True, help="Output path")
    parser.add_argument("-d", "--dim", type=int, help="Size of latent dimension (default: 10)", default=10)
    parser.add_argument("-m", "--max_iter", type=int, help="Max Iteration (default: 200)", default=30)
    args = parser.parse_args()

    # seed setting
    np.random.seed(0)

    # run pLSI
    theta = train_dense_pLSI(args.input_path, args.dim)
