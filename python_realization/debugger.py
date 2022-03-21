import sys
sys.path.append('./python_realization/includes')
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.linalg import norm
from math import sqrt, pow, ceil

from includes.constants import *
from includes.calculations import *
from includes.plotting import *

np.random.seed(42)

# Because temperature is an average kinetic energy of CHAOTIC movement, I'll need to substract
# the speed of center of mass from the speed of every atom to calculate the temperature

def get_start_hist_param(norm_vels, vels_x, vels_y, vels_z):
    '''
    The most important things here are starting edges, which are fixed for
    the whole averaging process
    '''
    heights_norm, edges_norm = np.histogram(norm_vels, bins=500)
    heights_x, edges_x = np.histogram(vels_x, bins=500)
    heights_y, edges_y = np.histogram(vels_y, bins=500)
    heights_z, edges_z = np.histogram(vels_z, bins=500)
    # we take only left edges here and then we'll use width
    return {
        'norm': (edges_norm[:-1], np.array(heights_norm)),
        'x': (edges_x[:-1], np.array(heights_x)),
        'y': (edges_y[:-1], np.array(heights_y)),
        'z': (edges_z[:-1], np.array(heights_z))
    }
    

def main_cycle(spawn_on_grid=True, sigma_for_vel=0.5):
    '''
    main cycle, all the movements and calculations will happen here
    '''
    particles = initialize_system(on_grid=spawn_on_grid, sigma_for_velocity=sigma_for_vel)
    total_pot = 0
    total_kin = 0
    #---
    energies = np.array([])
    kins = np.array([])
    pots = np.array([])
    steps_of_averaging = int(0.9 * TIME_STEPS)
    T_average = 0
    #---
    starting_hist_dic = get_start_hist_param(*achieve_velocities(particles))
    edges_norm = starting_hist_dic['norm'][0]
    edges_x = starting_hist_dic['x'][0]
    edges_y = starting_hist_dic['y'][0]
    edges_z = starting_hist_dic['z'][0]
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

        energies = np.append(energies, total_kin + total_pot)
        kins = np.append(kins, total_kin)
        pots = np.append(pots, total_pot)      
        T_current = (2 / 3) * (total_kin) / N

        # I will need to take averaged on time velocities to plot a histogram, so, let's do it:
        if ts >= TIME_STEPS - steps_of_averaging:
            list_of_velocities = achieve_velocities(particles)  

        #--------
        if ts % int((0.05 * TIME_STEPS)) == 0:
            print(f'{ts} steps passed, T_current = {T_current}')

    # let's start plotting:
    T_average /= steps_of_averaging
    

# ---------------------------------------- #

main_cycle(spawn_on_grid=True, sigma_for_vel=1.0)

# Усреднять количество элементов для каждого бина - сколько в среднем лежит в каждом бине - и получим ультра мега гладкую диаграмму

# При переходе через границу прибавляем длину ячейки - потому что сосденяя клетка точно такая же как наша и там частица движется точно так же
# смотрим сколько 