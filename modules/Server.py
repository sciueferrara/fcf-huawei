import random

random.seed(43)


class Server:
    def __init__(self, model, lr, fraction, positive_fraction, processing_strategy, send_strategy):
        self._processing_strategy = processing_strategy
        self._send_strategy = send_strategy
        self.model = model
        self.lr = lr
        self.fraction = fraction
        self.positive_fraction = positive_fraction

    def select_clients(self, clients, fraction=0.1):
        if fraction == 0:
            idx = random.sample(range(len(clients)), 1)
        else:
            idx = random.sample(range(len(clients)), int(fraction*len(clients)))
        return idx

    def train_on_client(self, clients, i):
        resulting_dic, resulting_bias = clients[i].train(self.lr, self.positive_fraction)
        for k, v in resulting_dic.items():
            self.model.item_vecs[k] += self.lr * v
        for k, v in resulting_bias.items():
            self.model.item_bias[k] += self.lr * v

    def train_model(self, clients):
        item_vecs_bak, item_bias_bak = self._send_strategy.backup_item_vectors(self.model) or (None, None)
        c_list = self.select_clients(clients, self.fraction)
        for i in c_list:
            self._send_strategy.send_item_vectors(clients, i, self.model)
        self._processing_strategy.train_model(self, clients, c_list)
        for i in c_list:
            self._send_strategy.delete_item_vectors(clients, i)
        self._send_strategy.update_deltas(self.model, item_vecs_bak, item_bias_bak)

    def predict(self, clients, max_k):
        predictions = []
        for i, c in enumerate(clients):
            self._send_strategy.send_item_vectors(clients, i, self.model)
            predictions.append(c.predict(max_k))
            self._send_strategy.delete_item_vectors(clients, i)
        return predictions
