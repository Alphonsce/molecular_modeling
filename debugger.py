import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.linalg import norm
from math import sqrt, pow
from constants import *

#141 seed was ok, 10 kinda
np.random.seed(10)

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

        self.kin_energy = 0     # 0.5 * M * norm(vel) ** 2
        self.pot_energy = 0

    def move(self):
        self.pos += self.vel * dt + 0.5 * self.acc * (dt ** 2)
        # boundary conditions:
        # do not work as intented

def initialize_system():
    '''
    initializes coordinates and velocities of particles
    '''
    f = open('trajectories.xyz', 'w')
    f1 = open('velocity.xyz', 'w')
    f2 = open('acceleration.xyz', 'w')
    f3 = open('momentums.xyz')
    particles = []
    if N == 2:
        for i in range(N):
            pos = np.array([100, 100, i * 1.1 + 100], dtype='float')        # 1.12 is a point of no force
            vel = np.zeros(3)
            acc = np.zeros(3)
            for k in range(3):
                vel[k] = random.normalvariate(0, 1)
            particles.append(Particle(pos, vel, acc))
        
    else:
        for i in range(N):
            pos = np.zeros(3)
            vel = np.zeros(3)
            acc = np.zeros(3)
            for k in range(3):
                pos[k] = np.random.uniform(0, L)
                vel[k] = random.normalvariate(0, 1)
            particles.append(Particle(pos, vel, acc))
            
    for i in range(N):
        for j in range(i + 1, N):
            calculate_acceleration(particles[i], particles[j])
    return particles

def force(r):
    '''
    r is a vector from one particle to another
    '''
    d = norm(r)
    f = 4 * (12 * pow(d, -13) - 6 * pow(d, -7)) * (r / d)   # wrong power of sigma is on purpose
    return f

def sgn(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0

def calculate_acceleration(part1, part2):
    r = part1.pos - part2.pos       # r_1 - r_2
    # Boundary condition realisation:
    for i in range(3):
        if abs(r[i]) > L / 2:
            r[i] = r[i] - L * sgn(r[i])
            
    dist = norm(r)
    if dist < r_cut:
        part1.acc += force(r) / M       # we add the force from only one particle acting on another to the total acc
        part2.acc -= force(r) / M
        # potential of two particle interaction, we need to add it to the total pot of one:
        part1.pot_energy += -4 * (pow(dist, -6) - pow(dist, -12))
        part2.pot_energy += 0

def check_boundary(particle):
    '''
    checks boundary conditions
    '''
    for i in range(3):
        if abs(particle.pos[i]) > L:
            particle.pos[i] %=  L
            print('123')


def plot_all_energies(energies, kin, pot):
    time = np.arange(0, len(energies) * dt, dt)
    plt.plot(time, energies, color='blue')
    plt.plot(time, kin, color='red')
    plt.plot(time, pot, color='green')
    plt.show()

def plot_total_energy(energies):
    time = np.arange(0, len(energies) * dt, dt)
    plt.plot(time, energies, color='blue')
    plt.show()

def plot_vel_distribution(velocities):
    plt.hist(velocities, N)
    plt.show()

def write_into_the_files(p):
    f.write('1 ' + str(p.pos[0]) + ' ' + str(p.pos[1]) + ' ' + str(p.pos[2]) + '\n')
    f1.write('1 ' + str(p.vel[0]) + ' ' + str(p.vel[1]) + ' ' + str(p.vel[2]) + '\n')
    f2.write('1 ' + str(p.acc[0]) + ' ' + str(p.acc[1]) + ' ' + str(p.acc[2]) + '\n')

def main_cycle():
    '''
    main cycle, all the movements and calculations will happen here
    '''
    particles = initialize_system()
    total_pot = 0
    total_kin = 0
    #---
    energies = np.array([])
    kins = np.array([])
    pots = np.array([])
    total_momentum = np.zeros(3)
    #---
    for ts in range(TIME_STEPS):
        f.write(str(N) + '\n')
        f.write('\n')
        #
        f1.write(str(N) + '\n')
        f1.write('\n')
        #
        f2.write(str(N) + '\n')
        f2.write('\n')
        #
        f3.write(str(total_momentum[0]) + ' ' + str(total_momentum[1]) + ' ' + str(total_momentum[2]) + '\n')
        f3.write('\n')
        #
        total_pot = 0
        total_kin = 0
        total_momentum = np.zeros(3)
        #--------------
        for p in particles:
            p.move()
            p.kin_energy = 0.5 * norm(p.vel) ** 2
            total_momentum += p.vel
            write_into_the_files(p)
            p.vel = p.vel + 0.5 * p.acc * dt # adding 1/2 * a(t) * dt
            p.acc = np.zeros(3)
            p.pot_energy = 0
            #check_boundary(p)
        for i in range(N):
            for j in range(i + 1, N):
                calculate_acceleration(particles[i], particles[j])

        for p in particles:
            total_kin += p.kin_energy
            total_pot += p.pot_energy

        energies = np.append(energies, total_kin + total_pot)
        kins = np.append(kins, total_kin)
        pots = np.append(pots, total_pot)
        
        T_current = (2 / 3) * total_kin / N
        print('Pot: ', total_pot, 'Kin: ', total_kin, 'Total: ', total_kin + total_pot)
        #--------
        for p in particles:
            p.vel += 0.5 * p.acc * dt   # adding 1/2 * a(t + dt)

    velocities = np.array([])
    for p in particles:
        velocities = np.append(velocities, norm(p.vel))
    #plot_vel_distribution(velocities)
    plot_all_energies(energies, kins, pots)
    #plot_total_energy(energies)

    print(T_current)

# ---------------------------------------- #

f = open('trajectories.xyz', 'r+')      #clearing a file
f.truncate(0)
#
f1 = open('velocity.xyz', 'r+')      #clearing a file
f1.truncate(0)
#
f2 = open('acceleration.xyz', 'r+')      #clearing a file
f2.truncate(0)
#
f3 = open('momentums.xyz', 'r+')      #clearing a file
f3.truncate(0)
main_cycle()

# Складывать сколько частиц попало в какой диапазон для N шагов, и потом делить на N