import torch
import random
from collections import deque
from torch.utils.data.sampler import Sampler

class Roller:
    def __init__(self, size, start_idx, order):
        self.start_idx = start_idx + [size]
        self.size = size
        self.order = order
        assert self.order < len(self.start_idx)

    def __next__(self):
        if self.n < self.size:
            self.n += 1
            # import ipdb
            # ipdb.set_trace()
            c = random.choice(self.available)
            self.available.remove(c)
            self.unavailable.append(c)
            self.available.append(self.unavailable.popleft())
            return random.randint(self.start_idx[c], self.start_idx[c+1]-1)
        else:
            raise StopIteration

    def __iter__(self):
        self.n = 0
        q = list(range(len(self.start_idx)-1))
        random.shuffle(q)
        self.unavailable = deque(q[:self.order])
        self.available = q[self.order:]
        return self

    def __len__(self):
        return self.size

class ScatteredSampler(Sampler):
    def __init__(self, data_source, start_idx, order, *args, **kwargs):
        super().__init__(data_source=data_source, *args, **kwargs)
        # data_source = sorted(data_source, key=lambda x: x[1])
        # assert all(data_source[i][1] <= data_source[i+1][1] for i in \
                   # range(len(data_source)-1))
        self.roller = Roller(len(data_source), start_idx, order)

    def __iter__(self):
        return iter(self.roller)

    def __len__(self):
        return self.roller.size
