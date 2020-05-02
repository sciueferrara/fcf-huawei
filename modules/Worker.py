import multiprocessing


class Worker(multiprocessing.Process):
    def __init__(self, task_queue, work, clients):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.work = work
        self.clients = clients

    def run(self):
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                self.task_queue.task_done()
                break
            self.work(self.clients, next_task)
            self.task_queue.task_done()
        return


class WorkerLocal(multiprocessing.Process):
    def __init__(self, task_queue, work, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.work = work
        self.result_queue = result_queue

    def run(self):
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                self.task_queue.task_done()
                break
            (i, j), lr, positive_fraction, bias_reg, user_reg, positive_item_reg, negative_item_reg = next_task
            self.work(i, j, lr, positive_fraction, bias_reg, user_reg, positive_item_reg, negative_item_reg)
            self.task_queue.task_done()
        return