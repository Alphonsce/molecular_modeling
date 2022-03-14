from turtle import position
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import iqr
from numpy.linalg import norm
from math import pow, ceil, sqrt
from constants import *

np.random.seed(42)

#arr = np.ones([TIME_STEPS, 3])
arr = np.array([[1, 2, 3], [7, 8, 9]])
arr = np.vstack(
    (arr, np.array([4, 5 , 6]))
    )
#arr = np.append(np.array([1, 2 , 3]))

print(arr)
arr_norms = np.array(list(map(lambda x: norm(x), arr)))
print(arr_norms)

print(np.std(arr))