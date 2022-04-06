import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from numpy.linalg import norm
from math import sqrt

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

def plot_ready_diffusion(path='../graphs/diffusion_ready/100p_300k_ready.csv'):
    df = pd.read_csv(path)
    dt_of_steps = df['dt_of_steps']
    all_means = df['all_means']

    plt.figure(figsize=(24, 16))
    sp = None

    plt.subplot(1, 2, 1)
    plt.scatter(dt_of_steps, all_means)
    plt.xlabel('$\Delta t$ of movement, $\sigma\cdot\sqrt{\dfrac{M}{\epsilon}}$', fontsize=14)
    plt.ylabel('$|\Delta r|^2$, $\sigma^2$', fontsize=14)

    x = np.log(dt_of_steps)
    y = np.log(all_means)
    plt.subplot(1, 2, 2)
    plt.scatter(x, y)
    plt.xlabel('$log(\Delta t$ of movement), $\sigma\cdot\sqrt{\dfrac{M}{\epsilon}}$', fontsize=14)
    plt.ylabel('$log(|\Delta r|^2)$, $\sigma^2$', fontsize=14)

    plt.show()


if __name__ == '__main__':
    # df1, Dt1 = make_df_get_Dt(path=path1, N=N)
    # df2, Dt2 = make_df_get_Dt(path=path2, N=N)

    df, Dt = make_df_get_Dt(path=path_best, N=N)
    all_means, dt_of_steps = calculate_all_means(max_step=22500, df=df, Dt=Dt, interval_for_step=225)


    plt.scatter(dt_of_steps, all_means)

    plt.show()