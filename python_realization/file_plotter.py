from inspect import _void
import matplotlib.pyplot as plt
# import modin.pandas as pd
import pandas as pd
import numpy as np
import sys

sys.path.append('./python_realization/includes')

from math import sqrt, pow
from numpy.linalg import norm

from diffusion_plotter import make_df_get_Dt, calculate_all_means, plot_ready_diffusion
from diffusion_plotter import convert_into_ready_diffusion, make_df_get_Dt

from includes.calculations import calculate_k

# У меня height - это сразу интеграл по ширине одной штучки ширины width, поскольку height - это вероятность нахождения
# частички в данном бине ширины width. Для получения плотность вероятности надо делать height / width

def make_ready_csv_diff(
    input_path='./graphs_rho_1_27/diffusion.csv',
    N=100, max_step=2380, interval_for_step=16,
    out_path='./graphs_rho_1_27/diffusion_ready/t_0_68_ready.csv'):
    '''
    Параметры по умолчанию стоят для файлов финального формата
    '''
    df, Dt = make_df_get_Dt(path=input_path, N=N)
    all_means, dt_of_steps = calculate_all_means(max_step=max_step, df=df, Dt=Dt, interval_for_step=interval_for_step)
    convert_into_ready_diffusion(dt_of_steps=dt_of_steps, all_means=all_means, out_path=out_path)

def plot_energies_from_file(path='./energies.csv', who_to_plot=['Total'], show=True):
    df = pd.read_csv(path)
    time = df.time
    for column in df.loc[:, who_to_plot]:
        plt.plot(time, df[column], label=column)
    plt.legend(loc='best', fontsize=14)
    plt.xlabel('Время, $\sigma\cdot\sqrt{\dfrac{M}{\epsilon}}$', fontsize=12)
    plt.ylabel('Энергия, $E/\epsilon$', fontsize=12)
    if show:
        plt.show()

def plot_hists_from_file(path='./histograms.csv', draw_gauss=True, show=True):
    '''
    Возвращает температуру, вычисленную по определению через кинетическую энергию
    '''
    df = pd.read_csv(path)
    heights = ['V_heights', 'V_x_heights', 'V_y_heights', 'V_z_heights']
    edges = ['V_edg', 'V_x_edg', 'V_y_edg', 'V_z_edg']
    names = ['$V$', '$V_x$', '$V_y$', '$V_z$']

    kT_avg = df['kT_average'][1]

    sb = None
    for i in range(len(names)):
        sb = plt.subplot(2, 2, 1 + i)
        width = 1. * (df[edges[i]][1] - df[edges[i]][0])
        if i == 0:
            if draw_gauss:
                x = np.linspace(0, max(df[edges[0]]), 1000)
                plt.plot(
                    x,
                    (1 / pow(2 * np.pi * kT_avg , 1.5)) * 4 * np.pi * (x ** 2) * np.exp( (-(x ** 2)) / (2 * kT_avg) ), color = 'red',
                )
            rho_prob = df[heights[i]] / width
            plt.bar(df[edges[i]], rho_prob, width, label=f'$kT={round(kT_avg, 3)} \epsilon$')
            plt.legend(loc='best', fontsize=12)
        else:
            rho_prob = df[heights[i]] / width
            plt.bar(df[edges[i]], rho_prob, width)
        plt.ylabel('$f_{probability}$', fontsize=14)
        plt.xlabel(names[i], fontsize=14)
    
    if show:
        plt.show()

    return round(kT_avg, 3)

def plot_gauss_lines_from_file(path='./gauss_lines.csv', train_part = 0.7, show=True):
    '''
    Возвращает температуру по аппроксимации Гаусса
    '''
    df = pd.read_csv(path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(axis=0)
    names = ['$V_x$', '$V_y$', '$V_z$']
    xs = ['V_x_edg_square', 'V_y_edg_square', 'V_z_edg_square']
    ys = ['log_V_x_heights', 'log_V_y_heights', 'log_V_z_heights']
    sp = None
    kT_avg = 0
    for i in range(len(names)):
        # width = abs(1. * (df[xs[i]][1] - df[xs[i]][0]))
        sb = plt.subplot(2, 2, i + 1)
        # df[ys[i]] -= np.log(width)     # getting \rho of proba from proba of being in a certain bin
        sub_df = df[[xs[i], ys[i]]]
        sub_df = sub_df.sort_values(by=xs[i])
        # Используем train_part датасета для "обучения" регрессии по МНК
        x_for_regression = np.array(list(sub_df[xs[i]])[:int(len(sub_df[xs[i]]) * train_part)])
        y_for_regression = np.array(list(sub_df[ys[i]])[:int(len(sub_df[xs[i]]) * train_part)])
        x_rest = np.array(list(sub_df[xs[i]])[int(len(sub_df[xs[i]]) * train_part): ])
        y_rest = np.array(list(sub_df[ys[i]])[int(len(sub_df[xs[i]]) * train_part): ])

        k, s_k, b, s_b = calculate_k(x_for_regression, y_for_regression)
        x = np.linspace(
            min(df[xs[i]]), max(df[xs[i]]), 1000
        )
        plt.plot(x, k * x + b, color='green', label='$-1/2\sigma^2$ = ' + str(round(-1 / (2 * k), 2)))
        plt.scatter(x_for_regression, y_for_regression, color='green')
        plt.scatter(x_rest, y_rest, color='red')
        plt.xlabel(names[i] + '$^2$')
        plt.ylabel('$ln(f_{prob})$')
        plt.title('Линеаризация по ' + names[i][-2].capitalize(), fontsize=10)
        plt.legend(loc='best')
        kT_avg += round(-1 / (2 * k), 4)

    if show:
        plt.show()

    return round(kT_avg / 3, 3)

def calculate_sigma(kT, D, rho=1 / 27):
    avg_V = np.sqrt(8 * (kT / np.pi))
    lamb = 3 * D / avg_V

    exp_section = 1 / (sqrt(2) * lamb * rho)      # "сечение" частицы, определенное через коэф диффузии
    exp_d = sqrt((4 / np.pi) * exp_section)

    theor_d = 1.34 * pow(kT, -1 / 12)       # "диаметр" частицы, определенный через теоритическую зависимость от температуры
    theor_section = np.pi * theor_d ** 2 / 4

    print(
        f'lambda = {round(lamb, 3)}; D = {D}; kT = {kT} \n',
        f'exp_section = {round(exp_section, 3)}; exp_d = {round(exp_d, 3)}; theor_section = {round(theor_section, 3)}; theor_d = {round(theor_d, 3)}',
        f'\n--------------------'
    )
    return

'''------------------------------------------------------------------------------------------------------------'''
''' ---- Плотность 1 / 27: ----'''
'''-----------------------------------------------------------------------------------------------------------'''

'''
T = 6.2; rho = 1 / 27: 
t=300_000 * dt, dt=0.0005, sigma_for_vel=2.5, bins_num=170, averaging_part=0.8, diffusion_step=50, N=100:
'''

# kT_def = plot_hists_from_file(path = './graphs_rho_1_27/histograms_6_2.csv')
# kT_gauss = plot_gauss_lines_from_file(path= './graphs_rho_1_27/lines_6_2.csv', train_part=0.75)
# plot_energies_from_file(who_to_plot=['Total'], path='./graphs_rho_1_27/energies_6_2.csv')
# D = plot_ready_diffusion(path='./graphs_rho_1_27/diffusion_ready/t_6_2_ready.csv')
# calculate_sigma(kT=kT_def, D=D)

'''
T = 4.7; rho = 1 / 27: 
t=300_000 * dt, dt=0.0005, sigma_for_vel=2.25, bins_num=170, averaging_part=0.8, diffusion_step=50, N=100:
'''

# kT_def = plot_hists_from_file(path = './graphs_rho_1_27/histograms_t_4_7.csv')
# kT_gauss = plot_gauss_lines_from_file(path= './graphs_rho_1_27/lines_t_4_7.csv', train_part=0.75)
# plot_energies_from_file(who_to_plot=['Total'], path='./graphs_rho_1_27/energies_t_4_7.csv')
# D = plot_ready_diffusion(path='./graphs_rho_1_27/diffusion_ready/t_4_7_ready.csv')
# calculate_sigma(kT=kT_def, D=D)

'''
T = 4.0; rho = 1 / 27: 
t=300_000 * dt, dt=0.0005, sigma_for_vel=2.0, bins_num=170, averaging_part=0.8, diffusion_step=50, N=100:
'''

# kT_def = plot_hists_from_file(path = './graphs_rho_1_27/histograms_t_4.csv')
# kT_gauss = plot_gauss_lines_from_file(path= './graphs_rho_1_27/lines_t_4.csv', train_part=0.75)
# plot_energies_from_file(who_to_plot=['Total'], path='./graphs_rho_1_27/energies_4.csv')
# D = plot_ready_diffusion(path='./graphs_rho_1_27/diffusion_ready/t_4_ready.csv')
# calculate_sigma(kT=kT_def, D=D)

'''
T = 3.2; rho = 1 / 27: 
t=300_000 * dt, dt=0.0005, sigma_for_vel=1.75, bins_num=170, averaging_part=0.8, diffusion_step=50, N=100:
'''

# kT_def = plot_hists_from_file(path = './graphs_rho_1_27/histograms3_2.csv')
# kT_gauss = plot_gauss_lines_from_file(path= './graphs_rho_1_27/lines3_2.csv', train_part=0.75)
# plot_energies_from_file(who_to_plot=['Total'], path='./graphs_rho_1_27/energies3_2.csv')
# D = plot_ready_diffusion(path='./graphs_rho_1_27/diffusion_ready/t_3_2_ready.csv')
# calculate_sigma(kT=kT_def, D=D)


'''
T = 2.3; rho = 1 / 27: 
t=300_000 * dt, dt=0.0005, sigma_for_vel=1.5, bins_num=170, averaging_part=0.8, diffusion_step=50, N=100:
'''

# kT_def = plot_hists_from_file(path = './graphs_rho_1_27/hists_2_3.csv')
# kT_gauss = plot_gauss_lines_from_file(path= './graphs_rho_1_27/lines_2_3.csv', train_part=0.75)
# plot_energies_from_file(who_to_plot=['Total'], path='./graphs_rho_1_27/energies_2_3.csv')
# D = plot_ready_diffusion(path='./graphs_rho_1_27/diffusion_ready/t_2_3_ready.csv')
# calculate_sigma(kT=kT_def, D=D)

'''
T = 1.65; rho = 1 / 27: 
t=300_000 * dt, dt=0.0005, sigma_for_vel=1.25, bins_num=170, averaging_part=0.8, diffusion_step=50, N=100:
'''

# kT_def = plot_hists_from_file(path = './graphs_rho_1_27/histograms_t_1_6.csv')
# kT_gauss = plot_gauss_lines_from_file(path= './graphs_rho_1_27/lines_t_1_6.csv', train_part=0.75)
# plot_energies_from_file(who_to_plot=['Total'], path='./graphs_rho_1_27/energies_t_1_6.csv')
# D = plot_ready_diffusion(path='./graphs_rho_1_27/diffusion_ready/t_1_6_ready.csv')
# calculate_sigma(kT=kT_def, D=D)

'''
T = 1.0; rho = 1 / 27: 
t=300_000 * dt, dt=0.0005, sigma_for_vel=1.0, bins_num=170, averaging_part=0.8, diffusion_step=50, N=100:
'''

# kT_def = plot_hists_from_file(path = './graphs_rho_1_27/histograms_t_1.csv')
# kT_gauss = plot_gauss_lines_from_file(path= './graphs_rho_1_27/gauss_lines_t_1.csv', train_part=0.75)
# plot_energies_from_file(who_to_plot=['Total'], path='./graphs_rho_1_27/energies_t_1.csv')
# D = plot_ready_diffusion(path='./graphs_rho_1_27/diffusion_ready/t_1_ready.csv')
# calculate_sigma(kT=kT_def, D=D)

'''
T = 0.68; rho = 1 / 27: 
t=300_000 * dt, dt=0.0005, sigma_for_vel=0.01, bins_num=170, averaging_part=0.8, diffusion_step=50, N=100:
'''

# kT_def = plot_hists_from_file(path = './graphs_rho_1_27/hists_0_68.csv')
# kT_gauss = plot_gauss_lines_from_file(path= './graphs_rho_1_27/lines_0_68.csv', train_part=0.75)
# plot_energies_from_file(who_to_plot=['Total'], path='./graphs_rho_1_27/energies_0_68.csv')
# D = plot_ready_diffusion(path='./graphs_rho_1_27/diffusion_ready/t_0_68_ready.csv')
# calculate_sigma(kT=kT_def, D=D)

'''------------------------------------------------------------------------------------------------------------'''
''' ---- Плотность 1 / 4: ----'''
'''-----------------------------------------------------------------------------------------------------------'''

'''
T = 4.6; rho = 1 / 4: 
t=300_000 * dt, dt=0.0005, sigma_for_vel=1.5, bins_num=170, averaging_part=0.8, diffusion_step=50, N=100:
'''

# kT_def = plot_hists_from_file(path = './graphs_rho_1_4/histograms_4_6.csv')
# kT_gauss = plot_gauss_lines_from_file(path= './graphs_rho_1_4/lines_4_6.csv', train_part=0.75)
# plot_energies_from_file(who_to_plot=['Total'], path='./graphs_rho_1_4/energies_4_6.csv')
# D = plot_ready_diffusion(path='./graphs_rho_1_4/diffusion_ready/t_4_6_ready.csv')
# calculate_sigma(kT=kT_def, D=D, rho=0.25)

'''
T = 2.7; rho = 1 / 4: 
t=300_000 * dt, dt=0.0005, sigma_for_vel=1.5, bins_num=170, averaging_part=0.8, diffusion_step=50, N=100:
'''

# kT_def = plot_hists_from_file(path = './graphs_rho_1_4/histograms_2_7.csv')
# kT_gauss = plot_gauss_lines_from_file(path= './graphs_rho_1_4/lines_2_7.csv', train_part=0.75)
# plot_energies_from_file(who_to_plot=['Total'], path='./graphs_rho_1_4/energies_2_7.csv')
# D = plot_ready_diffusion(path='./graphs_rho_1_4/diffusion_ready/t_2_7_ready.csv')
# calculate_sigma(kT=kT_def, D=D, rho=0.25)

'''
T = 1.5; rho = 1 / 4: 
t=300_000 * dt, dt=0.0005, sigma_for_vel=1.0, bins_num=170, averaging_part=0.8, diffusion_step=50, N=100:
'''

# kT_def = plot_hists_from_file(path = './graphs_rho_1_4/histograms_1_5.csv')
# kT_gauss = plot_gauss_lines_from_file(path= './graphs_rho_1_4/lines_1_5.csv', train_part=0.75)
# plot_energies_from_file(who_to_plot=['Total'], path='./graphs_rho_1_4/energies_1_5.csv')
# D = plot_ready_diffusion(path='./graphs_rho_1_4/diffusion_ready/t_1_5_ready.csv')
# calculate_sigma(kT=kT_def, D=D, rho=0.25)

''' For the last attempt: '''

# plot_hists_from_file()
# plot_gauss_lines_from_file()
# plot_energies_from_file(who_to_plot=['Total'])

# надо получить D(T) - потом уже тогда получим (sigma_exp / sigma_theor)[T]

# make_ready_csv_diff(
#     input_path='./graphs_rho_1_4/diffusion_4_6.csv',
#     N=100, max_step=2380, interval_for_step=16,
#     out_path='./graphs_rho_1_4/diffusion_ready/t_4_6_ready.csv')

