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

arr = [1, 2, 3, 4, 5, 6]

print(
    arr[-3:], arr[:3]
)