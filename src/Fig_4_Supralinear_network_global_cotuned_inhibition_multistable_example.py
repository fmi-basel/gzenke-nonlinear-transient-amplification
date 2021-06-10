import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sympy.solvers import solve
from sympy import Symbol
from matplotlib import patches
import matplotlib.patches as mpatches
import scipy.io as sio
from numpy import linalg as LA

# plotting configuration
ratio = 1.5
figure_len, figure_width = 15*ratio, 12*ratio
font_size_1, font_size_2 = 36*ratio, 36*ratio
legend_size = 18*ratio
line_width, tick_len = 3*ratio, 10*ratio
marker_size = 15*ratio
plot_line_width = 5*ratio
hfont = {'fontname': 'Arial'}

pal = sns.color_palette("deep")

# simulation setup
dt = 0.0001
T = int(16/dt)

# neuronal parameters
tau_e, tau_i = 0.020, 0.010

# network parameters
U, U_max = 1, 6
tau_x = 0.20

alpha_e, alpha_i = 2, 2

# network connectivity
l_l = [1]
k = 0.1

for l in l_l:
    Jee = 1.4
    Jie = (2-l) * 0.6
    Jei = (2-l) * 1.0
    Jii = (2-l) * 0.6

    Jee_2 = k * Jee
    Jie_2 = l * 0.6
    Jei_2 = l * 1.0
    Jii_2 = l * 0.6

    r_e_1, r_e_2, r_i_1, r_i_2 = 0, 0, 0, 0
    r_e = np.zeros(2)
    z_e_1, z_e_2, z_i_1, z_i_2 = 0, 0, 0, 0
    x = np.ones(2)
    l_r_e_1, l_r_e_2, l_r_i_1, l_r_i_2 = [], [], [], []

    for i in range(T):
        if 50000 <= i < 70000:
            g_e_1, g_e_2, g_i_1, g_i_2 = 2.2, 3.0, 2, 2
        elif 120000 <= i < 140000:
            g_e_1, g_e_2, g_i_1, g_i_2 = 3.0, 2.2, 2, 2
        else:
            g_e_1, g_e_2, g_i_1, g_i_2 = 2.2, 2.2, 2, 2

        g_e_1 = g_e_1 * (g_e_1 > 0)
        g_e_2 = g_e_2 * (g_e_2 > 0)
        g_i_1 = g_i_1 * (g_i_1 > 0)
        g_i_2 = g_i_2 * (g_i_2 > 0)

        # SSN part
        z_e_1 = Jee * r_e_1 + Jee_2 * r_e_2 - Jei * r_i_1 - Jei_2 * r_i_2 + g_e_1
        z_e_2 = Jee_2 * r_e_1 + Jee * r_e_2 - Jei_2 * r_i_1 - Jei * r_i_2 + g_e_2
        z_i_1 = Jie * x[0] * r_e_1 + Jie_2 * x[1] * r_e_2 - Jii * r_i_1 - Jii_2 * r_i_2 + g_i_1
        z_i_2 = Jie_2 * x[0] * r_e_1 + Jie * x[1] * r_e_2 - Jii_2 * r_i_1 - Jii * r_i_2 + g_i_2

        z_e_1 = z_e_1 * (z_e_1 > 0)
        z_e_2 = z_e_2 * (z_e_2 > 0)
        z_i_1 = z_i_1 * (z_i_1 > 0)
        z_i_2 = z_i_2 * (z_i_2 > 0)

        r_e_1 = r_e_1 + (-r_e_1 + np.power(z_e_1, alpha_e)) / tau_e * dt
        r_e_2 = r_e_2 + (-r_e_2 + np.power(z_e_2, alpha_e)) / tau_e * dt
        r_i_1 = r_i_1 + (-r_i_1 + np.power(z_i_1, alpha_i)) / tau_i * dt
        r_i_2 = r_i_2 + (-r_i_2 + np.power(z_i_2, alpha_i)) / tau_i * dt

        r_e_1 = r_e_1 * (r_e_1 > 0)
        r_e_2 = r_e_2 * (r_e_2 > 0)
        r_i_1 = r_i_1 * (r_i_1 > 0)
        r_i_2 = r_i_2 * (r_i_2 > 0)

        r_e[0] = np.copy(r_e_1)
        r_e[1] = np.copy(r_e_2)

        x = x + ((U - x) / tau_x + U * (U_max - x) * r_e) * dt
        x = x * (x > 0)
        x[x > U_max] = U_max

        l_r_e_1.append(r_e_1)
        l_r_e_2.append(r_e_2)
        l_r_i_1.append(r_i_1)
        l_r_i_2.append(r_i_2)

    l_r_e_1 = np.asarray(l_r_e_1)
    l_r_e_2 = np.asarray(l_r_e_2)
    l_r_i_1 = np.asarray(l_r_i_1)
    l_r_i_2 = np.asarray(l_r_i_2)

    if l == 1:
        print(np.mean(l_r_e_1[100000:110000]))
        print(np.mean(l_r_e_2[100000:110000]))
        print(np.mean(l_r_i_1[100000:110000]))
        print(np.mean(l_r_i_2[100000:110000]))

    plt.figure(figsize=(figure_len, figure_width))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)
    plt.yscale('symlog', linthreshy=0.1)

    plt.plot(l_r_e_1, linewidth=plot_line_width)
    plt.plot(l_r_e_2, linewidth=plot_line_width)
    plt.plot(l_r_i_1, linewidth=plot_line_width)
    # plt.plot(l_r_i_2, linewidth=plot_line_width)

    plt.xticks(np.arange(100000, 160000 + 5000, 20000), [0, 2, 4, 6], fontsize=font_size_1, **hfont)
    plt.yticks([0, 0.1, 1, 10, 100], fontsize=font_size_1, **hfont)
    plt.xlabel('Time (s)', fontsize=font_size_1, **hfont)
    plt.ylabel('Firing rate (Hz)', fontsize=font_size_1, **hfont)
    plt.xlim([100000, 160000])
    plt.ylim([0, 100])

    plt.legend(['Exc 1', 'Exc 2', 'Inh'], prop={"family": "Arial", 'size': font_size_1})

    if l == 1:
        plt.savefig('paper_figures/png/Fig_4_Supralinear_network_global_inhibition_multistable.png')
        plt.savefig('paper_figures/pdf/Fig_4_Supralinear_network_global_inhibition_multistable.pdf')
    else:
        plt.savefig('paper_figures/png/Fig_4_Supralinear_network_cotuned_inhibition_multistable.png')
        plt.savefig('paper_figures/pdf/Fig_4_Supralinear_network_cotuned_inhibition_multistable.pdf')