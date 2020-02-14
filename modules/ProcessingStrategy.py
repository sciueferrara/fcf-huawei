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
            resulting_dic, resulting_bias = clients[i].train(lr, with_pos)
            item_vecs_updates.append(resulting_dic)
            item_vecs_bias.append(resulting_bias)
        return item_vecs_updates, item_vecs_bias
