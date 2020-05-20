import multiprocessing
import numpy as np


class Worker(multiprocessing.Process):
    def __init__(self, task_queue, clients, shared_item_vecs, shared_item_vecs2, shape, lr):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.clients = clients
        self.shared_item_vecs = shared_item_vecs
        self.shared_item_vecs2 = np.frombuffer(shared_item_vecs2.get_obj()).reshape(self.shape)
        self.shape = shape
        self.lr = lr

    def run(self):
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                self.task_queue.task_done()
                break
            #with self.shared_item_vecs2.get_lock():
            self.shared_item_vecs2 += 1
            # resulting_dic = self.clients[next_task].train()
            # with self.shared_item_vecs.get_lock():
            #     new_item_vecs = np.frombuffer(self.shared_item_vecs.get_obj()).reshape(self.shape)
            #     for k, v in resulting_dic.items():
            #         new_item_vecs[k] += self.lr * 2 * v
            self.task_queue.task_done()
        return
