import random
from progress.bar import ChargingBar
import sys
import multiprocessing
from .Worker import Worker
import numpy as np

random.seed(43)


class Server:
    def __init__(self, model, lr, fraction, mp, send_strategy):
        self._send_strategy = send_strategy
        self.model = model
        self.lr = lr
        self.fraction = fraction
        self.progress = None
        self.mp = mp
        self.contatore = 0

    def select_clients(self, clients, fraction=0.1):
        if fraction == 0:
            idx = random.sample(range(len(clients)), 1)
        else:
            idx = random.sample(range(len(clients)), int(fraction*len(clients)))
        return idx

    def train_on_client(self, clients, i, prova):
        resulting_dic = clients[i].train()
        for k, v in resulting_dic.items():
            self.model.item_vecs[k] += self.lr * 2 * v
        #self.progress.next()
        self.contatore += 1
        #print(id(self.contatore))
        with prova.get_lock():
            prova.value += 1
            print(prova.value)
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
        self.progress = ChargingBar('Completing epoch', max=len(c_list), suffix="[%(index)d / %(remaining)d]")

        if not self.mp:
            for i in c_list:
                resulting_dic = clients[i].train()
                for k, v in resulting_dic.items():
                    self.model.item_vecs[k] += self.lr * 2 * v
                #self.train_on_client(clients, i)
        else:
            shared_item_vecs = multiprocessing.Array('d', self.model.item_vecs.size)
            item_vecs = np.frombuffer(shared_item_vecs.get_obj()).reshape(self.model.item_vecs.shape)[:] = self.model.item_vecs
            tasks = multiprocessing.JoinableQueue()
            num_workers = multiprocessing.cpu_count() - 1
            workers = [Worker(tasks, clients, shared_item_vecs, self.model.item_vecs.shape, self.lr) for _ in range(num_workers)]
            for w in workers:
                w.start()
            for i in c_list:
                tasks.put(i)
            for i in range(num_workers):
                tasks.put(None)
            tasks.join()
            self.model.item_vecs = item_vecs.copy()

            print(self.model.item_vecs[1])

        #self._processing_strategy.train_model(self, clients, c_list)
        self.model.item_vecs -= 2 * self.lr * regLambda * bak
        for i in c_list:
            self._send_strategy.delete_item_vectors(clients, i)
        self.progress = None
        self._send_strategy.update_deltas(self.model, item_vecs_bak, item_bias_bak)

    def predict(self, clients, max_k):
        predictions = []
        for i, c in enumerate(clients):
            self._send_strategy.send_item_vectors(clients, i, self.model)
            predictions.append(c.predict(max_k))
            self._send_strategy.delete_item_vectors(clients, i)
        return predictions
