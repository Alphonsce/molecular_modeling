import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from math import sqrt, pow, ceil
import pandas as pd

from constants import *
from calculations import *
# from gpu_calculations import *

#----------File stuff------------------
f_traj = open('trajectories.xyz', 'r+')
f_vel = open('velocity.xyz', 'r+')
FILES = [f_traj, f_vel]
for file in FILES:
    file.truncate(0)

def write_first_rows_in_files():
    for file in FILES:
        file.write(str(N) + '\n')
        file.write('\n')

def write_into_the_files(p, device='CPU'):
    if device=='CPU':
        for file in FILES:
            file.write('1 ' + str(p.pos[0]) + ' ' + str(p.pos[1]) + ' ' + str(p.pos[2]) + '\n')
    else:
        for file in FILES:
            v = p.pos.cpu().numpy()
            file.write('1 ' + str(v[0]) + ' ' + str(v[1]) + ' ' + str(v[2]) + '\n')

# ------------------------------------

def plot_all_energies(energies, kin, pot, show=True, output_path='./energies.csv'):
    time = np.arange(0, len(energies) * dt, dt)
    if show:
        plt.plot(time, energies, color='blue')
        plt.plot(time, kin, color='red')
        plt.plot(time, pot, color='green')
        plt.show()
    pd.DataFrame(
        {'time': time, 'Total': energies, 'Kin': kin, 'Pot': pot}
    ).to_csv(output_path, index=False)

def plot_total_energy(energies):
    time = np.arange(0, len(energies) * dt, dt)
    plt.plot(time, energies, color='blue')
    plt.show()

def get_start_hist_param(norm_vels, vels_x, vels_y, vels_z, kT, bin_number=50):
    '''
    The most important things here are starting edges, which are fixed for
    the whole averaging process,
    returning left edges for plotting a bar and all edges for calculating heights on the next steps
    '''

    heights_norm, edges_norm = np.histogram(norm_vels, bins=bin_number)
    heights_x, edges_x = np.histogram(vels_x, bins=bin_number)
    heights_y, edges_y = np.histogram(vels_y, bins=bin_number)
    heights_z, edges_z = np.histogram(vels_z, bins=bin_number)
    # readjusting edges for our Temperature, this is the edges, NOT the left edges for plotting:
    big_enough_vel = 3 * sqrt(3 * kT)
    two_thirds_of_big_enough_vel = 2 * sqrt(3 * kT)
    edges_norm = np.linspace(
        0, big_enough_vel, len(edges_norm)
    )
    edges_x = np.linspace(
        -two_thirds_of_big_enough_vel, two_thirds_of_big_enough_vel, len(edges_x)
    )
    edges_y = np.linspace(
        -two_thirds_of_big_enough_vel, two_thirds_of_big_enough_vel, len(edges_y)
    )
    edges_z = np.linspace(
        -two_thirds_of_big_enough_vel, two_thirds_of_big_enough_vel, len(edges_z)
    )
    # we take only left edges here and then we'll use width
    return {
        'norm': (edges_norm, np.array(heights_norm)),
        'x': (edges_x, np.array(heights_x)),
        'y': (edges_y, np.array(heights_y)),
        'z': (edges_z, np.array(heights_z))
    }

def get_hist_param(norm_vels, vels_x, vels_y, vels_z, edges_norm, edges_x, edges_y, edges_z):
    heights_norm, useless = np.histogram(norm_vels, bins=edges_norm)
    heights_x, useless = np.histogram(vels_x, bins=edges_x)
    heights_y, useless = np.histogram(vels_y, bins=edges_y)
    heights_z, useless = np.histogram(vels_z, bins=edges_z)

    return {
        'norm': np.array(heights_norm),
        'x': np.array(heights_x),
        'y': np.array(heights_y),
        'z': np.array(heights_z)
    }

def new_hist_plot(heights, edges, kT_avg, output_path='./histograms.csv', show=True):
    names = ['$V$', '$V_x$', '$V_y$', '$V_z$']
    dict_for_df = {}
    sb = None
    for i in range(len(heights)):
        sb = plt.subplot(2, 2, 1 + i)
        width = 0.95 * (edges[i][1] - edges[i][0])
        if i == 0:
            x = np.linspace(0, max(edges[0]), 1000)
            plt.plot(
                x,
                0.1 * (1 / pow(2 * np.pi * kT_avg , 1.5)) * 4 * np.pi * (x ** 2) * np.exp( (-(x ** 2)) / (2 * kT_avg) ), color = 'red',
            )
            plt.bar(edges[i][:-1], heights[i], width, label=f'$kT={round(kT_avg, 3)} \epsilon$')
            plt.legend(loc='best', fontsize=12)
        else:
            plt.bar(edges[i][:-1], heights[i], width)
        plt.ylabel('Процент частиц', fontsize=14)
        plt.xlabel(names[i], fontsize=14)

        x_name = names[i][1:-1] + '_heights'
        y_name = y_name = names[i][1:-1] + '_edg'
        dict_for_df[x_name] = heights[i]
        dict_for_df[y_name] = edges[i][:-1]
        
    if show:
        plt.show()
    # writing into the file: (I am writing left edges into the file)
    dict_for_df['kT_average'] = [kT_avg] * len(heights[0])
    df = pd.DataFrame(dict_for_df)
    df.to_csv(output_path, index=False)

def plot_gauss_lines(heights, edges, output_path='./gauss_lines.csv', show=True):
    names= ['$V_x$', '$V_y$', '$V_z$']
    dict_for_df = {}
    sp = None
    for i in range(len(heights)):
        sb = plt.subplot(2, 2, i + 1)
        x = np.array(edges[i][:-1]) ** 2
        y = np.log(heights[i])
        plt.scatter(x, y)
        plt.xlabel(names[i] + '$^2$')
        plt.ylabel('$ln($% частиц)')
        plt.title('Линеаризация распределения по ' + names[i])

        y_name = 'log_' + names[i][1:-1] + '_heights'
        x_name = names[i][1:-1] + '_edg_square'
        dict_for_df[y_name] = y
        dict_for_df[x_name] = x
    if show:
        plt.show()
    df = pd.DataFrame(dict_for_df)
    df.to_csv(output_path, index=False)

#--------------------Old version of hist plotting:--------------------------------------

# def plot_vel_distribution(vel_norms, vels_x, vels_y, vels_z, temperature, outpath='vels_plotting.csv'):
#     '''temperature in kT units, which are in epsilon units, so basically temprature = kT / epsilon'''
#     sigmas = []
#     mus = []
#     for arr in([vel_norms, vels_x, vels_y, vels_z]):
#         sigmas.append(np.std(arr))
#         mus.append(np.mean(arr))

#     sp = None
#     iter = 1
#     names = [r'$|V|$', r'$V_x$', r'$V_y$', r'$V_z$']
#     bin_size = int(round(pow(N, 0.65), 0))
#     for vels in [vel_norms, vels_x, vels_y, vels_z]:
#         sp = plt.subplot(2, 2, iter)
#         if iter != 1:
#             plt.hist(vels, bins=bin_size, label=f'$\sigma= ${round(sigmas[iter - 1], 2)} $\mu= ${round(mus[iter-1], 2)}', density=True)
#         else:
#             #x = np.arange(min(vel_norms), max(vel_norms), 0.01)
#             x = np.linspace(0, max(vel_norms), 1000)
#             plt.plot(
#                 x,
#                 (1 / pow(2 * np.pi * temperature , 1.5)) * 4 * np.pi * (x ** 2) * np.exp( (-(x ** 2)) / (2 * temperature) ), color = 'red',
#                 label='Распределение Максвелла при T'
#             )
#             plt.hist(vels, bins=bin_size, label=f'$kT={round(temperature, 3)} \epsilon$', density=True)
#         plt.ylabel('Число частиц', fontsize=14)
#         plt.xlabel(names[iter - 1], fontsize=14)
#         plt.grid(alpha=0.2)
#         plt.legend(loc='best', fontsize=11)

#         iter += 1
    
#     data = {'V': vel_norms, 'Vx': vels_x, 'Vy': vels_y, 'Vz': vels_z}
#     df = pd.DataFrame(data)
#     df.to_csv(outpath)

#     plt.show()