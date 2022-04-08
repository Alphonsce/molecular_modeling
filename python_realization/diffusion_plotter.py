import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('./python_realization/includes')

from numpy.linalg import norm
from math import sqrt

from includes.calculations import calculate_k
#---- diffusion plotting: ----

N = 100
# path1 = './graphs/diff_N100_t50_interval10dt.csv'
# path2 = './graphs/diff_N_100_dt0_00025.csv'
path_best = './graphs/diff_100p_300k_steps.csv'

def make_df_get_Dt(path, N):
    df = pd.read_csv(path, skiprows=[0,1, 2])

    with open(path) as f:
        f.readline()
        line = f.readline()

    Dt = float(line.split(':')[-1][:-1])    # шаг по времени для соседних строчек в диффузии

    return df, Dt

def calculate_all_means(max_step, df, Dt, interval_for_step=10):
    '''
    Возвращает массив из усредненного по всем перемещениям для каждой частицы и затем по всем частиц из перемещений для разных времен перемещения
    и массив отрезков времени для которых как раз получено значение перемещения.
    df: pd.Dataframe
    Dt: расстояние по времени между двумя соседними строчками в датафрейме
    '''

    steps = [step for step in range(1, max_step + 1, interval_for_step)]    # через такое количество строчек я смотрю перемещение-проходясь по циклу я для разных времен перемещения получаю значения
    # print(steps)
    
    all_means = []
    for step in steps:
        part_dict = {}
        for i in range(N):
            part_dict[i] = np.array([])
        
        for row_numb in steps:
            for p_numb in range(N):
                row = df.iloc[row_numb]
                x = row[str(p_numb) + 'x']
                y = row[str(p_numb) + 'y']
                z = row[str(p_numb) + 'z']
                pos = np.array([x, y, z])

                row_next = df.iloc[row_numb + step]     # берем для той же частицы с шагом step ряд
                x_next = row_next[str(p_numb) + 'x']
                y_next = row_next[str(p_numb) + 'y']
                z_next = row_next[str(p_numb) + 'z']
                pos_next = np.array([x_next, y_next, z_next])
                
                s_square = norm(pos_next - pos) ** 2
                part_dict[p_numb] = np.append(part_dict[p_numb], s_square)

        mean_dict = {}
        for i in range(N):
            mean_dict[i] = part_dict[i].mean()

        total_mean_for_step = np.array(list(mean_dict.values())).mean()

        all_means.append(total_mean_for_step)

    return all_means, Dt * np.array(steps)

def convert_into_ready_diffusion(all_means, dt_of_steps, out_path='./graphs/diffusion_ready/diff_ready'):
    pd.DataFrame(
        {'all_means': all_means, 'dt_of_steps': dt_of_steps}
    ).to_csv(out_path, index=False)

def plot_ready_diffusion(path='./graphs/diffusion_ready/100p_300k_ready.csv', n_approx=20):
    df = pd.read_csv(path)
    dt_of_steps = df['dt_of_steps']
    all_means = df['all_means']

    plt.figure(figsize=(12, 6))
    sp = None

    plt.subplot(1, 2, 1)
    plt.scatter(dt_of_steps, all_means)
    plt.xlabel('$\Delta t$ of movement, $\sigma\cdot\sqrt{\dfrac{M}{\epsilon}}$', fontsize=14)
    plt.ylabel('$|\Delta r|^2$, $\sigma^2$', fontsize=14)

    x = np.log(dt_of_steps)
    y = np.log(all_means)
    plt.subplot(1, 2, 2)

    # Построим прямую по МНК на первых n_approx и прямую на последних n_approx точках
    x_parab, y_parab = x[:n_approx], y[:n_approx]
    k_parab, _, b_parab, _ = calculate_k(x_parab, y_parab, through_zero=False)
    x_lin, y_lin = x[-n_approx:], y[-n_approx:]
    k_lin, _, b_lin, _ = calculate_k(x_lin, y_lin, through_zero=False)
    
    x_parab = x[:len(x) // 2]
    x_lin = x[int(-len(x) * 0.9):]

    plt.plot(x_parab, k_parab * x_parab + b_parab, color='red')
    plt.plot(x_lin, k_lin * x_lin + b_lin, color='orange')

    plt.scatter(x, y)
    plt.xlabel('$log(\Delta t$ of movement), $\sigma\cdot\sqrt{\dfrac{M}{\epsilon}}$', fontsize=14)
    plt.ylabel('$log(|\Delta r|^2)$, $\sigma^2$', fontsize=14)

    plt.show()

# Нужно просто построить прямую по первым точкам в первой прямой и по последним точкам во второй прямой,
# тогда точка их пересечения - по ох будет ln времени свободного пробега