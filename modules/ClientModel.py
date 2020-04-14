import numpy as np


class ClientModel:
    def __init__(self, n_items, n_factors):
        self.item_vecs = np.zeros((n_items, n_factors))
        self.item_bias = np.zeros(n_items)
        self.user_vec = np.random.randn(n_factors) / 10

    def predict(self):
        return np.dot(self.item_vecs, self.user_vec) + self.item_bias

    def predict_one(self, i):
        return np.dot(self.item_vecs[i], self.user_vec) + self.item_bias[i]
