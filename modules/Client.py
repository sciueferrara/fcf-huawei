import numpy as np
import random
import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg


class Client:
    def __init__(self, client_id, model, train, train_user_list, validation_user_list, test_user_list):
        self.id = client_id
        self.model = model
        self.train_user_list = train_user_list
        self.validation_user_list = validation_user_list
        self.test_user_list = test_user_list
        self.train_set = train
        self.Cu = sp.sparse.diags(train, 0)
        self.I = sp.sparse.diags(np.repeat(1, len(train)), 0)

    def predict(self, max_k):
        result = self.model.predict()
        result[list(self.train_user_list)] = -np.inf
        result = result.toarray().flatten()
        top_k = result.argsort()[-max_k:][::-1]
        top_k_score = result[top_k]
        prediction = {top_k[i]: top_k_score[i] for i in range(len(top_k))}

        return prediction

    def train(self):
        resulting_dic = {}
        regLambda = 0.1
        reg = sp.sparse.csr_matrix(regLambda * np.eye(self.model.item_vecs.shape[1]))

        Yt = self.model.item_vecs.T
        YtY = Yt.dot(self.model.item_vecs)

        YTCuY = YtY + Yt.dot(self.Cu - self.I).dot(self.model.item_vecs)
        self.model.user_vec = sp.sparse.csr_matrix(sp.sparse.linalg.spsolve(YTCuY + reg, Yt.dot(self.Cu).dot(sp.sparse.csr_matrix(np.ones(len(self.train_set))).T)))

        for i in range(len(self.train_set)):
            resulting_dic[i] = (sp.sparse.csr_matrix(self.train_set[i]) - self.model.user_vec.dot(self.model.item_vecs[i].T)) * self.model.user_vec

        return resulting_dic
