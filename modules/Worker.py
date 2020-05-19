import multiprocessing


class Worker(multiprocessing.Process):
    def __init__(self, task_queue, work, clients):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.work = work
        #self.clients = clients

    def run(self):
        while True:
            work, clients, next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                self.task_queue.task_done()
                break
            self.work(clients, next_task)
            #self.work(self.clients, next_task)
            self.task_queue.task_done()
        return
