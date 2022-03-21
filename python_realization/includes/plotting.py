import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from math import sqrt, pow, ceil
import pandas as pd

from constants import *
from calculations import *

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

def write_into_the_files(p):
    for file in FILES:
        file.write('1 ' + str(p.pos[0]) + ' ' + str(p.pos[1]) + ' ' + str(p.pos[2]) + '\n')

# ------------------------------------

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

def plot_vel_distribution(vel_norms, vels_x, vels_y, vels_z, temperature, outpath='vels_plotting.csv'):
    '''temperature in kT units, which are in epsilon units, so basically temprature = kT / epsilon'''
    sigmas = []
    mus = []
    for arr in([vel_norms, vels_x, vels_y, vels_z]):
        sigmas.append(np.std(arr))
        mus.append(np.mean(arr))

    sp = None
    iter = 1
    names = [r'$|V|$', r'$V_x$', r'$V_y$', r'$V_z$']
    bin_size = int(round(pow(N, 0.65), 0))
    for vels in [vel_norms, vels_x, vels_y, vels_z]:
        sp = plt.subplot(2, 2, iter)
        if iter != 1:
            plt.hist(vels, bins=bin_size, label=f'$\sigma= ${round(sigmas[iter - 1], 2)} $\mu= ${round(mus[iter-1], 2)}', density=True)
        else:
            #x = np.arange(min(vel_norms), max(vel_norms), 0.01)
            x = np.linspace(0, max(vel_norms), 1000)
            plt.plot(
                x,
                (1 / pow(2 * np.pi * temperature , 1.5)) * 4 * np.pi * (x ** 2) * np.exp( (-(x ** 2)) / (2 * temperature) ), color = 'red',
                label='Распределение Максвелла при T'
            )
            plt.hist(vels, bins=bin_size, label=f'$kT={round(temperature, 3)} \epsilon$', density=True)
        plt.ylabel('Число частиц', fontsize=14)
        plt.xlabel(names[iter - 1], fontsize=14)
        plt.grid(alpha=0.2)
        plt.legend(loc='best', fontsize=11)

        iter += 1
    
    data = {'V': vel_norms, 'Vx': vels_x, 'Vy': vels_y, 'Vz': vels_z}
    df = pd.DataFrame(data)
    df.to_csv(outpath)

    plt.show()