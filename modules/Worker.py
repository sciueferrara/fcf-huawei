import multiprocessing
import numpy as np
import scipy as sp
import scipy.sparse
import math

class Worker(multiprocessing.Process):
    def __init__(self, task_queue, shared_item_vecs, shape, lr, shared_counter, starting_model):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.shared_item_vecs = shared_item_vecs
        self.shared_counter = shared_counter
        self.shape = shape
        self.lr = lr
        self.starting_model = starting_model





    def run(self):
        while True:
            eps = 0.00000000001
            b1 = 0.05
            b2 = 0.05
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                self.task_queue.task_done()
                break

            clients, id = next_task
            with self.shared_counter.get_lock():
                self.shared_counter.value += 1
                print("Processing clients {} / {}\r".format(self.shared_counter.value, len(clients)), end="")

            regLambda = 1
            reg = sp.sparse.csr_matrix(regLambda * np.eye(self.starting_model.item_vecs.shape[1]))

            Yt = self.starting_model.item_vecs.T
            YtY = Yt.dot(self.starting_model.item_vecs)

            YTCuY = YtY + Yt.dot(clients[id].Cu - clients[id].I).dot(self.starting_model.item_vecs)
            calcolo = sp.sparse.csr_matrix(sp.sparse.linalg.spsolve(YTCuY + reg, Yt.dot(clients[id].Cu).dot(
                clients[id].train_set.T)))
            print(calcolo)
            clients[id].model.user_vec = calcolo
            print(clients[id].model.user_vec)

            grad = self.lr * 2 * (sp.sparse.csr_matrix(clients[id].train_set) -
                                  clients[id].model.user_vec * self.starting_model.item_vecs.T).T *\
                   clients[id].model.user_vec

            with self.shared_item_vecs.get_lock():
                item_vecs = np.frombuffer(self.shared_item_vecs.get_obj()).reshape(self.shape)
                item_vecs += grad
            self.task_queue.task_done()
        return
