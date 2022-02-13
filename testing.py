import numpy as np
import random
from numpy.linalg import norm
from math import pow
from constants import *

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

arr = np.array([1, 2, 3])
arr = np.append(arr, 5)
print(arr)

print(np.arange(0, TIME_STEPS * dt, dt))