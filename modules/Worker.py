import multiprocessing


class Worker(multiprocessing.Process):
    def __init__(self, task_queue, clients, shared_item_vecs, lr):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.clients = clients
        self.shared_item_vecs = shared_item_vecs
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
                for k, v in resulting_dic.items():
                    self.shared_item_vecs.item_vecs[k] += self.lr * 2 * v
            self.task_queue.task_done()
        return
