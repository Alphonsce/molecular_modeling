from turtle import position
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import iqr
from numpy.linalg import norm
from math import pow, ceil
from constants import *

np.random.seed(42)

class Particle:
    '''
    class, which will represent a moving particle
    '''
    def __init__(self, pos, vel, acc):
        self.pos = pos
        self.vel = vel
        self.acc = acc

        self.kin_energy = 0     # 0.5 * M * norm(vel) ** 2
        self.pot_energy = 0

# x надо менять раз в L // N, y надо менять раз в L ** 2 // N, z надо менять раз в L ** 3 // N 

def N_grid(n):
    return ceil(pow(n, 1 / 3))

N = 64
L = 4
print(N_grid(N))

for i in range(N):
    n_grid = N_grid(N)
    d = L / N_grid(N)
    vel = np.zeros(3)
    acc = np.zeros(3)
    x = d * (i % n_grid)
    y = d * (i // n_grid)
    z = d * (i // n_grid **  2)
    pos = np.array([x, y, z])
    print(pos)