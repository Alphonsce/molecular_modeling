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

ar = np.array([1, 3, 2])

print(sorted(ar))