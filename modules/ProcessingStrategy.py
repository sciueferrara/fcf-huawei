import abc
import multiprocessing
from .Worker import Worker


class ProcessingStrategy:
    @abc.abstractmethod
    def train_model(self, server, clients, c_list):
        pass


class SingleProcessing(ProcessingStrategy):
    def train_model(self, server, clients, c_list):
        for i in c_list:
            server.train_on_client(clients, i)


class MultiProcessing(ProcessingStrategy):
    def train_model(self, server, clients, c_list):
        prova = multiprocessing.Value('i', 0)
        tasks = multiprocessing.JoinableQueue()
        num_workers = multiprocessing.cpu_count() - 1
        workers = [Worker(tasks, server.train_on_client, clients, prova) for _ in range(num_workers)]
        for w in workers:
            w.start()
        for i in c_list:
            tasks.put(i)
        for i in range(num_workers):
            tasks.put(None)
        tasks.join()
