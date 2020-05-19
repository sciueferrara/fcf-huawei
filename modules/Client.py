import numpy as np
import random
import numpy as np
import scipy as sp


class Client:
    def __init__(self, client_id, model, train, train_user_list, validation_user_list, test_user_list):
        self.id = client_id
        self.model = model
        self.train_user_list = train_user_list
        self.validation_user_list = validation_user_list
        self.test_user_list = test_user_list
        self.train = train
        self.Cu = sp.spare.diags(train, 0)
        self.I = sp.sparse.diags(np.repeat(1, len(train)), 0)

    def predict(self, max_k):
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
        regLambda = 0.1
        reg = regLambda * np.eye(f, f)

        Yt = self.model.item_vecs.T
        YtY = Yt.dot(self.model.item_vecs)

        YTCuY = YtY + Yt.dot(self.Cu - self.I).dot(self.model.item_vecs)
        self.model.user_vec = sp.sparse.linalg.spsolve(YTCuY + reg, Yt.dot(self.Cu))

        for i in range(len(self.train)):
            resulting_dic[i] = (self.train[i] - self.model.user_vec.dot(self.model.item_vecs[i])) * self.model.user_vec
