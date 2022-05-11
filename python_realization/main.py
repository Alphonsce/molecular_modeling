import sys
sys.path.append('./python_realization/includes')
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.linalg import norm
from math import sqrt, pow, ceil

# import ray
# ray.init()

from includes.constants import *
from includes.calculations import *
# from includes.gpu_calculations import *
from includes.plotting import *

np.random.seed(42)

# Because temperature is an average kinetic energy of CHAOTIC movement, I'll need to substract
# the speed of center of mass from the speed of every atom to calculate the temperature

def main_cycle(spawn_on_grid=True, sigma_for_vel=0.5, verbose=1, bins_num=50, averaging_part=0.8, diffusion_step=1, device='CPU', rho_coef=2):
    '''
    main cycle, all the movements and calculations will happen here
    verbose: % of program finished to print
    diffusion_step: once every diffusion_step all coordinates will be written for diffusion plotting
    rho = 1 / rho_coef ** 3
    '''
    particles = initialize_system(on_grid=spawn_on_grid, sigma_for_velocity=sigma_for_vel, device=device)
    total_pot = 0
    total_kin = 0
    #---
    energies = np.array([])
    kins = np.array([])
    pots = np.array([])
    #---
    steps_of_averaging = int(averaging_part * TIME_STEPS)
    T_average = 0
    heights_norm_avg = np.array([])
    heights_x_avg = np.array([])
    heights_y_avg = np.array([])
    heights_z_avg = np.array([])
    #---
    diffusion_writer = write_step_of_diffusion_and_create_writer(diffusion_step=diffusion_step, path='diffusion.csv')
    #---
    for ts in range(TIME_STEPS):
        write_first_rows_in_files()
        total_pot = 0
        total_kin = 0
        #-----moving---------
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
        #---

        energies = np.append(energies, total_kin + total_pot)
        kins = np.append(kins, total_kin)
        pots = np.append(pots, total_pot)      
        T_current = (2 / 3) * (total_kin) / N

        # Starting things for a set conditions:
        if ts >= TIME_STEPS - steps_of_averaging:
            if ts ==  TIME_STEPS - steps_of_averaging:
                starting_hist_dic = get_start_hist_param(*achieve_velocities(particles), T_current, bin_number=bins_num)
                edges_norm, heights_norm_avg = starting_hist_dic['norm']
                edges_x, heights_x_avg = starting_hist_dic['x']
                edges_y, heights_y_avg = starting_hist_dic['y']
                edges_z, heights_z_avg = starting_hist_dic['z']

            hist_dic = get_hist_param(*achieve_velocities(particles), edges_norm, edges_x, edges_y, edges_z)
            heights_norm_avg += hist_dic['norm']
            heights_x_avg += hist_dic['x']
            heights_y_avg += hist_dic['y']
            heights_z_avg += hist_dic['z']
            T_average += T_current
            for p in particles:
                p.diffusion_move()
            if (ts - steps_of_averaging) % diffusion_step == 0:
                write_diffusion(diffusion_writer, particles, time=ts * dt)
        #--------
        if int((0.01 * verbose * TIME_STEPS)) != 0:
            if ts % int((0.01 * verbose * TIME_STEPS)) == 0:
                print(f'{ts} steps passed, T_current = {T_current}')
        else:
            print(f'{ts} steps passed, T_current = {T_current}')

    # let's start plotting:
    T_average /= steps_of_averaging
    n = N * 1
    heights = [
        heights_norm_avg / (steps_of_averaging * n), 
        heights_x_avg / (steps_of_averaging * n),
        heights_y_avg / (steps_of_averaging * n),
        heights_z_avg / (steps_of_averaging * n)
    ]
    edges = [edges_norm, edges_x, edges_y, edges_z]
    new_hist_plot(heights, edges, T_average, show=False)
    plot_gauss_lines(heights[1:], edges[1:], show=False)
    plot_all_energies(energies, kins, pots, show=False)

# ---------------------------------------- #
if __name__ == '__main__':
    main_cycle(spawn_on_grid=True, sigma_for_vel=2.0, bins_num=170, averaging_part=0.8, diffusion_step=50)
