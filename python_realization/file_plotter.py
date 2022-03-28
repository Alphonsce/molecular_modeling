import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import sqrt

def calculate_k(x, y, through_zero=False):
    '''Вычисление коэффициентов для аппроксимации зависимостью y = kx + b'''
    n = len(x)
    m_x = x.mean()
    m_y = y.mean()
    m_xx = (x * x). mean()
    m_yy = (y * y).mean()
    m_xy = (x * y).mean()
    m_x_m_x = x.mean() * x.mean()
    m_y_m_y = y.mean() * y.mean()

    k = 0
    s_k = 0
    b = 0
    s_b = 0

    if through_zero:
        k = m_xy / m_xx
        s_k = (1 / sqrt(n)) * sqrt((m_yy / m_xx) - k ** 2)
        return [k, s_k]

    else:
        k = (m_xy - m_x * m_y) / (m_xx - m_x_m_x)
        b = m_y - k * m_x

        s_k = (1 / sqrt(n)) * sqrt((m_yy - m_y_m_y) / (m_xx - m_x_m_x) - k ** 2)
        s_b = s_k * sqrt(m_xx - m_x_m_x)
        return [k, s_k, b, s_b]

def plot_energies_from_file(path='./energies.csv', who_to_plot=['Total']):
    df = pd.read_csv(path)
    time = df.time
    for column in df.loc[:, who_to_plot]:
        plt.plot(time, df[column], label=column)
    plt.legend(loc='best', fontsize=14)
    plt.xlabel('Время, $\sigma\cdot\sqrt{\dfrac{M}{\epsilon}}$', fontsize=12)
    plt.ylabel('Энергия, $E/\epsilon$', fontsize=12)
    plt.show()

def plot_hists_from_file(path='./histograms.csv', draw_gauss=True):
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
                    0.1 * (1 / pow(2 * np.pi * kT_avg , 1.5)) * 4 * np.pi * (x ** 2) * np.exp( (-(x ** 2)) / (2 * kT_avg) ), color = 'red',
                )
            plt.bar(df[edges[i]], df[heights[i]], width, label=f'$kT={round(kT_avg, 3)} \epsilon$')
            plt.legend(loc='best', fontsize=12)
        else:
            plt.bar(df[edges[i]], df[heights[i]], width)
        plt.ylabel('Процент частиц', fontsize=14)
        plt.xlabel(names[i], fontsize=14)
    plt.show()
    h = (df['V_x_heights']).sum()
    print(h)

def plot_gauss_lines_from_file(path='./gauss_lines.csv', train_part = 0.7):
    df = pd.read_csv(path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(axis=0)
    names = ['$V_x$', '$V_y$', '$V_z$']
    ys = ['log_V_x_heights', 'log_V_y_heights', 'log_V_z_heights']
    xs = ['V_x_edg_square', 'V_y_edg_square', 'V_z_edg_square']
    sp = None
    for i in range(len(names)):
        sb = plt.subplot(2, 2, i + 1)
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
        plt.ylabel('$ln($% частиц)')
        plt.title('Линеаризация распределения по ' + names[i][-2].capitalize(), fontsize=10)
        plt.legend(loc='best')
    plt.show()

# For t=20, dt=0.0005, N=150, sigma_v = 5:

plot_hists_from_file(path='./graphs/histograms_20t_150p.csv')
plot_gauss_lines_from_file(path='./graphs/gauss_lines_20t_150p.csv')
# plot_energies_from_file(who_to_plot=['Total'], path='./graphs/energies_20t_150p.csv')

# For t=50, dt=0.0005, N=100, sigma_v = 1.5:

# plot_hists_from_file(draw_gauss=True, path = './graphs/histograms_100k_steps_100particles_dt_0_0005.csv')
# plot_gauss_lines_from_file(path = './graphs/lines_100k_steps_100particles_dt_0_0005.csv')
# plot_energies_from_file(who_to_plot=['Total'], path = './graphs/energies_100k_steps_100particles_dt_0_0005.csv')