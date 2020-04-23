import abc
import numpy as np


class SendStrategy:
    @abc.abstractmethod
    def broadcast_item_vectors(self, clients, model):
        pass

    @abc.abstractmethod
    def send_item_vectors(self, clients, i, model):
        pass

    @abc.abstractmethod
    def backup_item_vectors(self, model):
        pass

    @abc.abstractmethod
    def update_deltas(self, model, item_vecs_bak, item_bias_bak):
        pass

    @abc.abstractmethod
    def delete_item_vectors(self, clients, i):
        pass


class SendVector(SendStrategy):
    def broadcast_item_vectors(self, clients, model):
        for i, c in enumerate(clients):
            c.model.item_vecs = np.copy(model.item_vecs)
            c.model.item_bias = np.copy(model.item_bias)

    def send_item_vectors(self, clients, i, model):
        clients[i].model.item_vecs = np.copy(model.item_vecs)
        clients[i].model.item_bias = np.copy(model.item_bias)

    def backup_item_vectors(self, model):
        pass

    def update_deltas(self, model, item_vecs_bak, item_bias_bak):
        pass

    def delete_item_vectors(self, clients, i):
        del clients[i].model.item_vecs
        del clients[i].model.item_bias


class SendDelta(SendStrategy):
    def broadcast_item_vectors(self, clients, model):
        for i, c in enumerate(clients):
            c.model.item_vecs += model.item_vecs_delta
            c.model.item_bias += model.item_bias_delta

    def send_item_vectors(self, clients, i, model):
        clients[i].model.item_vecs += model.item_vecs_delta
        clients[i].model.item_bias += model.item_bias_delta

    def backup_item_vectors(self, model):
        item_vecs_bak = np.copy(model.item_vecs)
        item_bias_bak = np.copy(model.item_bias)
        return item_vecs_bak, item_bias_bak

    def update_deltas(self, model, item_vecs_bak, item_bias_bak):
        model.item_vecs_delta = model.item_vecs - item_vecs_bak
        model.item_bias_delta = model.item_bias - item_bias_bak

    def delete_item_vectors(self, clients, i):
        del clients[i].model.item_vecs
        del clients[i].model.item_bias
