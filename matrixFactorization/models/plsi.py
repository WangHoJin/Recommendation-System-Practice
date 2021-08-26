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
        # self.pZDW : [n_topics, n_docs, n_vocs]        
        # self.dw_matrix : [n_docs, n_vocs]
        observed = np.zeros((self.nDocuments, self.nVocabulary, self.nTopics))
        for doc_id in range(self.nDocuments):
            # self. pZDW[:, doc_id, :] : [n_topics, n_vocs]             
            # self.dw_matrix[doc_id, :] : [n_vocs]            
            # self.dw_matrix[doc_id, :].reshape(1, -1) : [1, n_vocs]             
            # use broadcast. [n_topics, n_vocs] x [1, n_vocs] => [n_topics, n_vocs] x [n_topic, n_vocs] 
            # [n_topics, n_vocs] x [n_topics. n_vocs] => [n_topics, n_vocs]
            # [n_topics, n_vocs].T => [n_vocs, n_topics]
            observed[doc_id] = (self.pZDW[:, doc_id, :] * self.dw_matrix[doc_id, :].reshape(1, -1)).T
        return observed

    def update_pZD(self, observed):
        # pZD: [n_topics, n_docs]        
        # observed : [n_docs, n_vocs, n_topics]        
        self.pZD = observed.sum(axis=1).T  # [n_topic, n_doc]
        self.pZD /= self.dw_matrix.sum(axis=1).reshape(1, self.nDocuments)               

    def update_pWZ(self, observed):
        # pWZ: [n_vocs, n_topics]
        # observed : [n_docs, n_vocs, n_topics]
        self.pWZ = observed.sum(axis=0)
        
        # topic-wise normalization (easy)
        for topic_id in range(self.nTopics):
            self.pWZ[:, topic_id] /= self.pWZ[:, topic_id].sum() # since 1-d, we don't need to specify axis
        
        # parallel normalization (difficult)
        # self.pWZ : [n_vocs, n_topics]
        # self.pWZ.sum(axis=0).reshape(1, -1) : [1, n_topics]
        # self.pWZ /= self.pWZ.sum(axis=0).reshape(1, -1)
 
    def update_pZDW(self):
        # update pZDW : [n_topics, n_docs, n_vocs]        
        # pWZ: [n_vocs, n_topics]        
        # pZD: [n_topics, n_docs]
        
	# Hard solution
        a = self.pZD.reshape(self.nTopics, self.nDocuments, 1)  # [n_topics, n_docs, 1]        
        b = self.pWZ.T.reshape(self.nTopics, 1, self.nVocabulary)  # # [n_topics, 1, n_vocs]
        self.pZDW = a * b
        self.pZDW /= self.pZDW.sum(axis=0).reshape(1, self.nDocuments, self.nVocabulary)

	# Simpler solution
        #for topic_id in range(self.nTopics):
        #    self.pZDW[topic_id, :, :]  = self.pWZ[:, topic_id].reshape(1, -1) * self.pZD[topic_id, :].reshape(-1, 1)  # # [1, n_vocs] x [n_docs, 1]	


        # self.pZDW: [nTopic, nDoc, nVoca]        
        #for word_id in range(self.nVocabulary):
        #    for doc_id in range(self.nDocuments):
        #        sum_dw = self.pZDW[:, doc_id, word_id].sum()
        #        for topic_id in range(self.nTopics):
        #              self.pZDW[topic_id, doc_id, word_id] /= sum_dw

    def compute_log_likelihood_slowest(self):
        log_likelihood = 0.
        for doc_id in range(self.nDocuments):
            for word_id in range(self.nVocabulary):
                tmp = self.dw_matrix[doc_id, word_id] * np.log(1./self.nDocuments * (self.pZD[:, doc_id] * self.pWZ[word_id, :]).sum())
                log_likelihood += tmp
        return log_likelihood
    

    def compute_log_likelihood_fast(self):
        # Fast version.
        log_likelihood = 0.
        for doc_id in range(self.nDocuments):
            tmp = self.dw_matrix[doc_id, :] * np.log(1./self.nDocuments * (self.pZD[:, doc_id].reshape(1, -1) * self.pWZ[:, :]).sum(axis=1))
            log_likelihood += tmp.sum()
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
    item_ids = pickle.load(open(input_path + '/item_ids.pkl', 'rb'))

    file = open("../ml_plot.dat", encoding="utf-8").readlines()
    document_mapping = dict()
    d_id = 0

    word_mapping = dict()
    w_id = 0

    for row in file:
        index = row.find("::")
        if row[:index] in item_ids:
            document_mapping[int(row[:index])] = d_id
            d_id += 1
            for word in row[index + 2:-2].replace("|", " ").split():
                if word not in word_mapping:
                    word_mapping[word] = w_id
                    w_id += 1
    dw_matrix = np.zeros((d_id, w_id))
    d_iter = 0
    for row in file:
        index = row.find("::")
        if row[:index] in item_ids:
            for word in row[index + 2:-2].replace("|", " ").split():
                dw_matrix[d_iter][word_mapping[word]] += 1
            d_iter += 1
    return dw_matrix

def gen_document_word_frequency_sparse(input_path):
    item_ids = pickle.load(open(input_path + '/item_ids.pkl', 'rb'))

    file = open("ml_plot.dat", encoding="utf-8").readlines()
    document_mapping = dict()
    d_id = 0

    word_mapping = dict()
    w_id = 0

    for row in file:
        index = row.find("::")
        if row[:index] in item_ids:
            document_mapping[int(row[:index])] = d_id
            d_id += 1
            for word in row[index + 2:-2].replace("|", " ").split():
                if word not in word_mapping:
                    word_mapping[word] = w_id
                    w_id += 1
    dw_matrix = np.zeros((d_id, w_id))
    d_iter = 0
    for row in file:
        index = row.find("::")
        if row[:index] in item_ids:
            for word in row[index + 2:-2].replace("|", " ").split():
                dw_matrix[d_iter][word_mapping[word]] += 1
            d_iter += 1

    # Sparsification of data
    sparse_data, non_zero_idxs = [], []
    counts = 0
    for document_id in range(dw_matrix.shape[0]):
        non_zero_idx = dw_matrix[document_id, :].nonzero()
        t = dw_matrix[document_id, non_zero_idx]
        counts += t.shape[1]

        sparse_data.append(t[0])
        non_zero_idxs.append(non_zero_idx[0])
        # print("non_zero_idx", non_zero_idx[0].shape)
    return sparse_data, non_zero_idxs

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
