from turtle import position
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import iqr
from numpy.linalg import norm
from math import pow
from constants import *

np.random.seed(12)

pos = np.array([1, 2, 3])

for i in range(3):
    pos[i] = 1

print(pos)