import random
from progress.bar import ChargingBar
import sys

random.seed(43)


class Server:
    def __init__(self, model, lr, fraction, processing_strategy, send_strategy):
        self._processing_strategy = processing_strategy
        self._send_strategy = send_strategy
        self.model = model
        self.lr = lr
        self.fraction = fraction

    def select_clients(self, clients, fraction=0.1):
        if fraction == 0:
            idx = random.sample(range(len(clients)), 1)
        else:
            idx = random.sample(range(len(clients)), int(fraction*len(clients)))
        return idx

    def train_on_client(self, clients, i, progress):
        resulting_dic = clients[i].train()
        for k, v in resulting_dic.items():
            self.model.item_vecs[k] += self.lr * 2 * v
        progress.next()
        #for k, v in resulting_bias.items():
        #    self.model.item_bias[k] += self.lr * v

    def train_model(self, clients):
        regLambda = 0.1
        bak = self.model.item_vecs.copy()
        item_vecs_bak, item_bias_bak = self._send_strategy.backup_item_vectors(self.model) or (None, None)
        c_list = self.select_clients(clients, self.fraction)
        for i in c_list:
            self._send_strategy.send_item_vectors(clients, i, self.model)
        sys.stdout.write('\x1b[1A')
        sys.stdout.flush()
        progress = ChargingBar('Completing epoch', max=len(c_list), suffix="[%(index)d / %(remaining)d]")
        self._processing_strategy.train_model(self, clients, c_list, progress)
        self.model.item_vecs -= self.lr * regLambda * bak
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
