import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.linalg import norm
from math import sqrt, pow, ceil
from constants import *

np.random.seed(42)

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
        check_boundary(self)

def N_grid(n):
    return ceil(pow(n, 1 / 3))

def initialize_system(on_grid=False, sigma_for_vel=0.5):
    '''
    initializes coordinates and velocities of particles
    '''
    f = open('trajectories.xyz', 'w')
    f1 = open('velocity.xyz', 'w')
    f2 = open('acceleration.xyz', 'w')
    particles = []
    
    for i in range(N):
        vel = np.zeros(3)
        acc = np.zeros(3)
        pos = np.zeros(3)
        if on_grid:
            n_grid = N_grid(N)
            d = L / N_grid(N)
            vel = np.zeros(3)
            acc = np.zeros(3)
            x = d * (i % n_grid) + np.random.uniform(-d / 20, d / 20)
            y = d * (i // n_grid) + np.random.uniform(-d / 20, d / 20)
            z = d * (i // n_grid **  2) + np.random.uniform(-d / 20, d / 20)
            pos = np.array([x, y, z])
            for k in range(3):
                vel[k] = random.normalvariate(0, 1)
        elif not on_grid:
            for k in range(3):
                pos[k] = np.random.uniform(0, L)
                vel[k] = random.normalvariate(0, 0)
        particles.append(Particle(pos, vel, acc))
    # calculation of initialized accelerations:
    for i in range(N):
        for j in range(i + 1, N):
            calculate_acceleration(particles[i], particles[j])
    return particles

def force(r):
    '''
    r is a vector from one particle to another
    '''
    d = norm(r)
    f = 4 * (12 * pow(d, -13) - 6 * pow(d, -7)) * (r / d)
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
        # potential of two particle interaction, we need to add it to the total pot of one: Both have a half of the total
        part1.pot_energy += -4 * (pow(dist, -6) - pow(dist, -12))
        part2.pot_energy += 0

def check_boundary(particle):
    '''
    checks boundary conditions
    '''
    for i in range(3):
        if particle.pos[i] > L:
            particle.pos[i] -=  L
        elif particle.pos[i] < 0:
            particle.pos[i] += L

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

def plot_vel_distribution(particles):
    '''
    particles : array of N particles
    velocities : 2d array of 1d arrays-vectors of velocity at the final moment for every particle
    '''
    velocities = particles[0].vel
    for p_number in range(1, N):
        velocities = np.vstack(
            (velocities, particles[p_number].vel)
        )
    
    vel_norms = np.array(list(map(lambda x: norm(x), velocities)))

    velocities_x = np.array([])
    velocities_y = np.array([])
    velocities_z = np.array([])
    for v in velocities:
        velocities_x = np.append(velocities_x ,v[0])
        velocities_y = np.append(velocities_y ,v[1])
        velocities_z = np.append(velocities_z ,v[2])

    sp = None
    iter = 1
    names = [r'$V$', r'$V_x$', r'$V_y$', r'$V_z$']
    bin_size = int(round(pow(N, 0.65), 0))
    for vels in [vel_norms, velocities_x, velocities_y, velocities_z]:
        sp = plt.subplot(2, 2, iter)
        plt.hist(vels, bins=bin_size)
        plt.ylabel('Число частиц', fontsize=14)
        plt.xlabel(names[iter - 1], fontsize=14)
        plt.grid(alpha=0.2)

        iter += 1

    plt.show()

def write_first_rows_in_files():
    f.write(str(N) + '\n')
    f.write('\n')
    #
    f1.write(str(N) + '\n')
    f1.write('\n')
    #
    f2.write(str(N) + '\n')
    f2.write('\n')

def write_into_the_files(p):
    f.write('1 ' + str(p.pos[0]) + ' ' + str(p.pos[1]) + ' ' + str(p.pos[2]) + '\n')
    f1.write('1 ' + str(p.vel[0]) + ' ' + str(p.vel[1]) + ' ' + str(p.vel[2]) + '\n')
    f2.write('1 ' + str(p.acc[0]) + ' ' + str(p.acc[1]) + ' ' + str(p.acc[2]) + '\n')

def main_cycle(spawn_on_grid=True):
    '''
    main cycle, all the movements and calculations will happen here
    '''
    particles = initialize_system(on_grid=spawn_on_grid)
    total_pot = 0
    total_kin = 0
    #---
    energies = np.array([])
    kins = np.array([])
    pots = np.array([])
    #---
    for ts in range(TIME_STEPS):
        write_first_rows_in_files()
        #
        total_pot = 0
        total_kin = 0
        #--------------
        for p in particles:
            p.move()
            p.kin_energy = 0.5 * norm(p.vel) ** 2
            write_into_the_files(p)
            p.vel = p.vel + 0.5 * p.acc * dt # adding 1/2 * a(t) * dt
            p.acc = np.zeros(3)
            p.pot_energy = 0
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
        print('Step number: ' + str(ts), 'Pot: ', total_pot, 'Kin: ', total_kin, 'Total: ', total_kin + total_pot)
        #--------
        for p in particles:
            p.vel += 0.5 * p.acc * dt   # adding 1/2 * a(t + dt)

    plot_all_energies(energies, kins, pots)
    plot_total_energy(energies)
    plot_vel_distribution(particles)

    print(T_current)

# ---------------------------------------- #

main_cycle(spawn_on_grid=True)

# Складывать сколько частиц попало в какой диапазон для N шагов, и потом делить на N