import numpy as np

np.random.seed(43)


class TripletSampler:
    def __init__(self, train_user_list, item_size, sampler_size):
        self.train_user_list = list(train_user_list)
        self.item_size = item_size
        self.sampler_size = sampler_size

    def sample_user_triples(self):
        for _ in range(20):
            i = np.random.choice(self.train_user_list)
            j = np.random.randint(self.item_size)
            while j in self.train_user_list:
                j = np.random.randint(self.item_size)
            yield i, j
