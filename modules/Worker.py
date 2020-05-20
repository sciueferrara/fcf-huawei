import multiprocessing


class Worker(multiprocessing.Process):
    def __init__(self, task_queue, clients, prova):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.clients = clients
        self.prova = prova

    def run(self):
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                self.task_queue.task_done()
                break
            #self.work(self.clients, next_task, self.prova)
            self.prova.value += 1
            self.task_queue.task_done()
        return
