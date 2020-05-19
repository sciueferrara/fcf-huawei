import numpy as np
import scipy as sp
import scipy.sparse


class ServerModel:
    def __init__(self, n_items, n_factors):
        self.item_vecs = sp.sparse.csr_matrix(np.random.randn(n_items, n_factors)/10)
        self.item_bias = np.random.randn(n_items)/10
        self.item_vecs_delta = np.copy(self.item_vecs)
        self.item_bias_delta = np.copy(self.item_bias)
