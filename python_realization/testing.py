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
import csv

from includes.constants import *
from includes.calculations import N_grid

np.random.seed(42)

f = open('diffusion.csv', 'w')
fieldnames = ['col1', 'col2']
writer = csv.DictWriter(f, fieldnames=fieldnames)
f.write('slkhj' + '\n')

writer.writeheader()
writer.writerow(
    {
    'col1': 1,
    'col2': 2
    }
)

print(writer.fieldnames)