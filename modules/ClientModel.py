import numpy as np
import scipy as sp
import scipy.sparse

class ClientModel:
    def __init__(self, n_factors):
        self.item_vecs = None
        self.item_bias = None
        self.user_vec = sp.sparse.csr_matrix(np.random.randn(n_factors) / 10)

    def predict(self):
        return np.dot(self.item_vecs, self.user_vec) + self.item_bias

    def predict_one(self, i):
        return np.dot(self.item_vecs[i], self.user_vec) + self.item_bias[i]
