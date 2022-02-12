import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.linalg import norm
from math import sqrt, pow
from constants import *

np.random.seed(1)

# Because temperature is an average kinetic energy of CHAOTIC movement, I'll need to substract
# the speed of center of mass from the speed of every atom to calculate the temperature

class Particle:
    '''
    class, which will represent a moving particle
    '''
    def __init__(self, pos, vel, acc):
        self.pos = pos
        self.vel = vel
        self.acc = acc

        self.energy = 0.5 * norm(vel) ** 2

    def move(self):
        pass

def initialize_system():
    '''
    initializes coordinates and velocities of particles
    '''
    particles = []
    for _ in range(N):
        pos = np.zeros(3)
        vel = np.zeros(3)
        acc = np.zeros(3)
        for i in range(3):
            pos[i] = random.uniform(0, L)
            vel[i] = random.normalvariate(0, 1)
        particles.append(Particle(pos, vel, acc))
    return particles

def force(r):
    '''
    r is a vector from one particle to another
    '''
    d = norm(r)
    f = 4 * EPSILON * (12 * (SIGMA / pow(d, 13)) - 6 * (SIGMA / pow(d, 7))) * (r / d)
    return f

def sgn(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0

def calculate_acceleration(part1, part2):
    r = part1.pos - part2.pos       # delta r -
    # Boundary condition realisation:
    for i in range(3):
        if r[i] > L / 2:
            r[i] = r[i] - L * sgn(r[i])
            
    dist = norm(r)
    if dist < r_cut:
        part1.acc = force(r) / M
        part2.acc = -part1.acc

def main_function():
    '''
    main cycle, all the movements and calculations will happen here
    '''
    particles = initialize_system()

    for ts in range(TIME_STEPS):
        pass

# ---------------------------------------- #
particles = initialize_system()

for i in range(1, N):
    f_vec = force(particles[0].pos - particles[i].pos)
    print(f_vec)