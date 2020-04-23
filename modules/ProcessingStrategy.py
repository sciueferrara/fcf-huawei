import abc


class ProcessingStrategy:
    @abc.abstractmethod
    def train_model(self, clients, c_list, lr, with_pos):
        pass


class SingleProcessing(ProcessingStrategy):
    def train_model(self, clients, c_list, lr, with_pos):
        item_vecs_updates = []
        item_vecs_bias = []
        for i in c_list:
            self._send_strategy.send_item_vectors(clients, c_list, self.model)
            resulting_dic, resulting_bias = clients[i].train(lr, with_pos)
            for k, v in resulting_dic.items():
                self.model.item_vecs[k] += self.lr * v
            for k, v in resulting_bias.items():
                self.model.item_bias[k] += self.lr * v
            self._send_strategy.delete_item_vectors(clients, [i])
        return item_vecs_updates, item_vecs_bias
