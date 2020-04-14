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

    def select_clients_and_send(self, clients, fraction=0.1):
        if fraction == 0:
            idx = random.sample(range(len(clients)), 1)
        else:
            idx = random.sample(range(len(clients)), int(fraction*len(clients)))
        self._send_strategy.send_item_vectors(clients, idx, self.model)
        return idx

    def train_model(self, clients):
        item_vecs_bak, item_bias_bak = self._send_strategy.backup_item_vectors(self.model) or (None, None)
        c_list = self.select_clients_and_send(clients, self.fraction)
        item_update_list, bias_update_list = self._processing_strategy.train_model(clients, c_list, self.lr, self.positive_fraction)
        for i in item_update_list:
            for k, v in i.items():
                self.model.item_vecs[k] += self.lr * v
        for i in bias_update_list:
            for k, v in i.items():
                self.model.item_bias[k] += self.lr * v
        self._send_strategy.update_deltas(self.model, item_vecs_bak, item_bias_bak)

    def predict(self, clients, max_k):
        self._send_strategy.broadcast_item_vectors(clients, self.model)
        return [c.predict(max_k) for c in clients]
