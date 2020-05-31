import multiprocessing
import numpy as np
import scipy as sp
import scipy.sparse
import math

class Worker(multiprocessing.Process):
    def __init__(self, task_queue, clients, shared_item_vecs, shared_user_vecs, shape, shape_uv, lr, shared_counter, starting_model):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.shared_item_vecs = shared_item_vecs
        self.shared_user_vecs = shared_user_vecs
        self.shared_counter = shared_counter
        self.shape = shape
        self.lr = lr
        self.starting_model = starting_model
        self.clients = clients
        self.shape_uv = shape_uv





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

            with self.shared_counter.get_lock():
                self.shared_counter.value += 1
                print("Processing clients {} / {}\r".format(self.shared_counter.value, len(self.clients)), end="")

            regLambda = 1
            reg = sp.sparse.csr_matrix(regLambda * np.eye(self.starting_model.item_vecs.shape[1]))

            Yt = self.starting_model.item_vecs.T
            YtY = Yt.dot(self.starting_model.item_vecs)

            YTCuY = YtY + Yt.dot(self.clients[next_task].Cu - self.clients[next_task].I).dot(self.starting_model.item_vecs)
            calcolo = sp.sparse.csr_matrix(sp.sparse.linalg.spsolve(YTCuY + reg, Yt.dot(self.clients[next_task].Cu).dot(
                self.clients[next_task].train_set.T)))
            print(calcolo)

            with self.shared_item_vecs.get_lock():
                user_vecs = np.frombuffer(self.shared_user_vecs.get_obj()).reshape(self.shape_uv)
                user_vecs[next_task] = calcolo
                print(self.shared_user_vecs[next_task])

            grad = self.lr * 2 * (sp.sparse.csr_matrix(self.clients[next_task].train_set) -
                                  self.shared_user_vecs[next_task] * self.starting_model.item_vecs.T).T *\
                   self.shared_user_vecs[next_task]

            with self.shared_item_vecs.get_lock():
                item_vecs = np.frombuffer(self.shared_item_vecs.get_obj()).reshape(self.shape)
                item_vecs += grad
            self.task_queue.task_done()
        return
