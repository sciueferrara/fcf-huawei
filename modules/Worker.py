import multiprocessing
import numpy as np
import scipy as sp
import scipy.sparse


class Worker(multiprocessing.Process):
    def __init__(self, task_queue, clients, shared_item_vecs, shape, lr, shared_counter, starting_model):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.clients = clients
        self.shared_item_vecs = shared_item_vecs
        self.shared_counter = shared_counter
        self.shape = shape
        self.lr = lr
        self.starting_model = starting_model

    def run(self):
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                self.task_queue.task_done()
                break
            regLambda = 0.1
            reg = sp.sparse.csr_matrix(regLambda * np.eye(self.starting_model.item_vecs.shape[1]))

            Yt = self.starting_model.item_vecs.T
            YtY = Yt.dot(self.starting_model.item_vecs)

            YTCuY = YtY + Yt.dot(self.clients[next_task].Cu - self.clients[next_task].I).dot(self.starting_model.item_vecs)
            self.clients[next_task].model.user_vec = sp.sparse.csr_matrix(sp.sparse.linalg.spsolve(YTCuY + reg, Yt.dot(self.clients[next_task].Cu).dot(
                sp.sparse.csr_matrix(np.ones(len(self.clients[next_task].train_set))).T)))


            resulting_dic = self.clients[next_task].train()
            with self.shared_item_vecs.get_lock():
                item_vecs = np.frombuffer(self.shared_item_vecs.get_obj()).reshape(self.shape)
                for i in range(len(self.clients[next_task].train_set)):
                    item_vecs[i] += self.lr * 2 * (
                            sp.sparse.csr_matrix(self.clients[next_task].train_set[i]) - self.clients[next_task].model.user_vec.dot(
                        self.starting_model.item_vecs[i].T)) * self.clients[next_task].model.user_vec
            with self.shared_counter.get_lock():
                self.shared_counter.value += 1
                print("Processing clients {} / {}\r".format(self.shared_counter.value, len(self.clients)), end="")
            self.task_queue.task_done()
        return
