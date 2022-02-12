import numpy as np
import matplotlib.pyplot as plt

SIGMA = 1
EPSILON = 1
M = 1

TIME_STEPS = 100
N = 20
L = N ** (1 / 3)

dt = 0.001
#rho = 
#T = 

class Particle:
    def __init__(self, pos, vel):
        self.pos = pos
        self.vel = vel
        self.acc = 0

    def move(self):
        pass

# def initialize_particles():
#     for _ in range(N):

def main_function():
    '''это главный цикл, в котором будет происходить расчет взаимодействия частиц и их движение
    '''
    pass
