import numpy as np
import random
import sys
sys.path.append('./python_realization/includes')
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import iqr
from numpy.linalg import norm
from math import pow, ceil, sqrt
import seaborn as sns

from includes.constants import *
from includes.calculations import N_grid

np.random.seed(42)

# edg = np.histogram_bin_edges(df.V, 50)      # ищет границы бинов по переданному ей распределению параметра

# Все что надо - это научиться строить гистограмму по заданынм counts и edges
n_grid = N_grid(N)
d = L / n_grid
# print(n_grid)
for k in range(N // n_grid ** 2):
    for j in range(N // n_grid):
        for i in range(N % n_grid):
            x = i * d
            y = j * d
            z = k * d
            print(f'{i, j, k}: {x, y, z}')

for i in range(N):
    x = i % n_grid
    z = i // n_grid ** 2
    y = i // n_grid - n_grid * z

    print(f'{i}: {x, y, z}')