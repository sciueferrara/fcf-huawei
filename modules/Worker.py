import multiprocessing
import numpy as np


class Worker(multiprocessing.Process):
    def __init__(self, task_queue, clients, shared_item_vecs, shape, lr, shared_counter):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.clients = clients
        self.shared_item_vecs = shared_item_vecs
        self.shared_counter = shared_counter
        self.shape = shape
        self.lr = lr

    def run(self):
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                self.task_queue.task_done()
                break
            resulting_dic = self.clients[next_task].train()
            with self.shared_item_vecs.get_lock():
                item_vecs = np.frombuffer(self.shared_item_vecs.get_obj()).reshape(self.shape)
                for k, v in resulting_dic.items():
                    item_vecs[k] += self.lr * 2 * v
            with self.shared_counter.get_lock():
                self.shared_counter.value += 1
                print("Processing clients {} / {}\r".format(self.shared_counter.value, len(self.clients)), end="")
            self.task_queue.task_done()
        return
