import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.linalg import norm
from math import sqrt
from constants import *

class Particle:
    '''
    class, which will represent a moving particle
    '''
    def __init__(self, pos, vel):
        self.pos = pos
        self.vel = vel
        self.acc = 0

        self.energy = norm(vel)

    def move(self):
        pass

def initialize_system():
    '''
    initializes coordinates and velocities of particles
    with a uniform distribution
    '''
    for _ in range(N):
        pos = np.zeros(3)

def recalculate_forces():
    '''
    recalculates forces for every particle
    '''
    pass

def main_function():
    '''
    main cycle, all the movements and calculations will happen here
    '''
    pass
