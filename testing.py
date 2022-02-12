import numpy as np
import random
from numpy.linalg import norm

np.random.RandomState(12)

arr = np.array([1, 2, 3])
arr = np.append(arr, [i for i in range(1, 10) if i % 2 == 0])

arr = np.array(['a', 'b', 'c', 5])

class Sheep:
    def __init__(self, size):
        self.size = size

sheeps = np.array([Sheep(size=i) for i in range(5)])

mapped_sheeps = np.array(list(map(lambda x: x.size, sheeps)))

# ----------------------------------- #

print(random.normalvariate(0, 1))

m = abs(np.array([2, 0]))
print(norm(m))