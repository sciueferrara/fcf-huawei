import numpy as np
from time import time

class Client:
    def __init__(self, client_id, model, train, train_user_list, validation_user_list, test_user_list):
        self.id = client_id
        self.model = model
        self.train_set = train
        self.train_user_list = train_user_list
        self.validation_user_list = validation_user_list
        self.test_user_list = test_user_list

    def test(self, max_k):
        result = self.model.predict()
        result[list(self.train_user_list)] = -np.inf
        top_k = result.argsort()[-max_k:][::-1]
        top_k_score = result[top_k]
        prediction = {top_k[i]: top_k_score[i] for i in range(len(top_k))}

        return prediction

    def train(self, lr, positive_fraction):
        bias_reg = 0
        user_reg = lr / 20
        positive_item_reg = lr / 20
        negative_item_reg = lr / 200
        resulting_dic = {}
        resulting_bias = {}

        for i, j in self.train_set.sample_user_triples():
            x_i = self.model.predict_one(i)
            x_j = self.model.predict_one(j)
            x_ij = x_i - x_j

            d_loss = 1 / (1 + np.exp(x_ij))

            bi = self.model.item_bias[i].copy()
            bj = self.model.item_bias[j].copy()

            self.model.item_bias[i] += lr * (d_loss - bias_reg * bi)
            self.model.item_bias[j] += lr * (-d_loss - bias_reg * bj)

            wu = self.model.user_vec.copy()
            hi = self.model.item_vecs[i].copy()
            hj = self.model.item_vecs[j].copy()

            self.model.user_vec += lr * (d_loss * (hi - hj) - user_reg * wu)
            self.model.item_vecs[i] += lr * (d_loss * wu - positive_item_reg * hi)
            self.model.item_vecs[j] += lr * (d_loss * (-wu) - negative_item_reg * hj)

            resulting_dic[j] = d_loss * (-wu) - negative_item_reg * hj
            resulting_bias[j] = -d_loss - bias_reg * bj
            if positive_fraction:
                if np.random.choice([True, False], p=[positive_fraction, 1-positive_fraction]):
                    resulting_dic[i] = d_loss * wu - positive_item_reg * hi
                    resulting_bias[i] = d_loss - bias_reg * bi

        return resulting_dic, resulting_bias
