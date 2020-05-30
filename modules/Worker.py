import multiprocessing
import numpy as np
import scipy as sp
import scipy.sparse
import math

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
            eps = 0.00000000001
            b1 = 0.05
            b2 = 0.05
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

            print('inizio')
            #print(len(self.clients[next_task].train_set))
            #for i in range(len(self.clients[next_task].train_set)):
            grad = self.lr * 2 * (
                    sp.sparse.csr_matrix(self.clients[next_task].train_set[i]) - self.clients[next_task].model.user_vec.dot(
                self.starting_model.item_vecs[i].T)) * self.clients[next_task].model.user_vec
            print('fatto')

                # self.clients[next_task].m = b1 * self.clients[next_task].m + (1 - b1) * grad
                # mhat = self.clients[next_task].m / (1 - b1)
                # self.clients[next_task].v = b2 * self.clients[next_task].v + (1 - b2) * grad.power(2)
                # vhat = self.clients[next_task].v / (1 - b2)

            with self.shared_item_vecs.get_lock():
                item_vecs = np.frombuffer(self.shared_item_vecs.get_obj()).reshape(self.shape)
                item_vecs[i] += grad
            with self.shared_counter.get_lock():
                self.shared_counter.value += 1
                print("Processing clients {} / {}\r".format(self.shared_counter.value, len(self.clients)), end="")
            self.task_queue.task_done()
        return
