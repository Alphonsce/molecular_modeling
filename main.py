from importlib.metadata import files
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.linalg import norm
from math import sqrt, pow, ceil
from constants import *

np.random.seed(42)

# Because temperature is an average kinetic energy of CHAOTIC movement, I'll need to substract
# the speed of center of mass from the speed of every atom to calculate the temperature

f_traj = open('trajectories.xyz', 'r+')
f_vel = open('velocity.xyz', 'r+')
FILES = [f_traj, f_vel]
for file in FILES:
    file.truncate(0)

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

def achieve_velocities(particles):
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

    return [vel_norms, velocities_x, velocities_y, velocities_z]

def plot_vel_distribution(vel_norms, vels_x, vels_y, vels_z, temperature):
    sigmas = []
    mus = []
    for arr in([vel_norms, vels_x, vels_y, vels_z]):
        sigmas.append(np.std(arr))
        mus.append(np.mean(arr))

    sp = None
    iter = 1
    names = [r'$V$', r'$V_x$', r'$V_y$', r'$V_z$']
    bin_size = int(round(pow(N, 0.65), 0))
    for vels in [vel_norms, vels_x, vels_y, vels_z]:
        sp = plt.subplot(2, 2, iter)
        if iter != 1:
            plt.hist(vels, bins=bin_size, label=f'$\sigma= ${round(sigmas[iter - 1], 2)} $\mu= ${round(mus[iter-1], 2)}')
        else:
             plt.hist(vels, bins=bin_size, label=f'$T={round(temperature, 3)}$')
        plt.ylabel('Число частиц', fontsize=14)
        plt.xlabel(names[iter - 1], fontsize=14)
        plt.grid(alpha=0.2)
        plt.legend(loc='best', fontsize=11)

        iter += 1

    plt.show()

def write_first_rows_in_files():
    for file in FILES:
        file.write(str(N) + '\n')
        file.write('\n')

def write_into_the_files(p):
    for file in FILES:
        file.write('1 ' + str(p.pos[0]) + ' ' + str(p.pos[1]) + ' ' + str(p.pos[2]) + '\n')

def calculate_com_vel(particles):
    Vc = np.zeros(3)
    for p in particles:
        for i in range(3):
            Vc[i] += p.vel[i]
    Vc /= N
    return Vc

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
    vels_for_plotting = [np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)]
    steps_of_averaging = 0.9 * TIME_STEPS
    #---
    for ts in range(TIME_STEPS):
        write_first_rows_in_files()
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
            p.vel += 0.5 * p.acc * dt   # adding 1/2 * a(t + dt)

        # I will need to take averaged on time velocities to plot a histogram, so, let's do it:
        if ts >= 0.1 * TIME_STEPS:
            list_of_velocities = achieve_velocities(particles)
            for arr_num in range(len(vels_for_plotting)):
                vels_for_plotting[arr_num] += list_of_velocities[arr_num]

        energies = np.append(energies, total_kin + total_pot)
        kins = np.append(kins, total_kin)
        pots = np.append(pots, total_pot)
        
        Vc = norm(calculate_com_vel(particles))
        T_current = (2 / 3) * (total_kin - 0.5 * Vc ** 2) / N       # subsracting velocity of COM
        #--------
        
        print('Step number: ' + str(ts), 'Pot: ', total_pot, 'Kin: ', total_kin, 'Total: ', total_kin + total_pot)

    # let's start plotting:
    plot_all_energies(energies, kins, pots)
    plot_total_energy(energies)
    for arr in vels_for_plotting:
        arr /= steps_of_averaging
    plot_vel_distribution(*vels_for_plotting, T_current)

# ---------------------------------------- #

main_cycle(spawn_on_grid=True)

# Складывать сколько частиц попало в какой диапазон для N шагов, и потом делить на N