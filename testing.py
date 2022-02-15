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
    if i == 2:
        pos[i] -= 1


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

part = Particle(np.array([7, 5, 8]), np.array([0.2, 0.1, 0.3]), np.zeros(3))

for i in range(50):
    print(np.random.uniform(0, L))

def move(self):
    self.pos += self.vel * dt + 0.5 * self.acc * (dt ** 2)
    # boundary conditions:
    # do not work as intented
    for i in range(3):
        if self.pos[i] > L:
            self.pos[i] -= L
        if self.pos[i] < L:
            self.pos[i] += L

print(15000.1298717 % 5)