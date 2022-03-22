from turtle import position
from nbformat import read
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import iqr
from numpy.linalg import norm
from math import pow, ceil, sqrt
import seaborn as sns

from includes.constants import *

np.random.seed(42)

# edg = np.histogram_bin_edges(df.V, 50)      # ищет границы бинов по переданному ей распределению параметра

# Все что надо - это научиться строить гистограмму по заданынм counts и edges

df = pd.read_csv('vels_last_step.csv')
df1 = pd.read_csv('vels_average.csv')

heights, edges = np.histogram(df.V, bins=500)
heights1, useless = np.histogram(df1.V, bins=edges)

print(len(edges))

heights += heights1
heights = np.array(heights) / 10000

left_edges = np.linspace(0, 3.5, len(heights))
width = 0.95 * (left_edges[1] - left_edges[0])
plt.bar(left_edges, heights, width, align='edge')

plt.show()