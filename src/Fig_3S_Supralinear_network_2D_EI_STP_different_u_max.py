import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sympy.solvers import solve
from sympy import Symbol
from matplotlib import patches
import matplotlib.patches as mpatches
import scipy.io as sio

# plotting configuration
ratio = 1.5
figure_len, figure_width = 15*ratio, 12*ratio
font_size_1, font_size_2 = 36*ratio, 36*ratio
legend_size = 18*ratio
line_width, tick_len = 3*ratio, 10*ratio
marker_size = 15*ratio
plot_line_width = 5*ratio
marker_edge_width = 4
hfont = {'fontname': 'Arial'}

# simulation setup
dt = 0.0001
T = int(9/dt)

# neuronal parameters
tau_e, tau_i = 0.020, 0.010
alpha_e, alpha_i = 2, 2

# adaptation
U = 1
tau_x = 0.2
x = 1

# network connectivity
Jee = 1.8
Jie = 1.0
Jei = 1.0
Jii = 0.6

l_U_max = [4, 5, 6]

l_peak, l_ss = [], []

for U_max in l_U_max:
    r_e, r_i = 0, 0
    z_e, z_i = 0, 0

    l_r_e, l_r_i = [], []

    x = 1

    for i in range(T):
        if 50000 <= i < 70000:
            g_e, g_i = 3.0, 2
        else:
            g_e, g_i = 1.55, 2

        g_e = g_e * (g_e > 0)
        g_i = g_i * (g_i > 0)

        # SSN part
        z_e = Jee * r_e - Jei * r_i + g_e
        z_i = Jie * x * r_e - Jii * r_i + g_i

        z_e = z_e * (z_e > 0)
        z_i = z_i * (z_i > 0)

        r_e = r_e + (-r_e + np.power(z_e, alpha_e)) / tau_e * dt
        r_i = r_i + (-r_i + np.power(z_i, alpha_i)) / tau_i * dt

        r_e = r_e * (r_e > 0)
        r_i = r_i * (r_i > 0)

        # adaptation of excitatory neurons
        x = x + ((U - x) / tau_x + U * (U_max - x) * r_e) * dt
        x = np.clip(x, 0, U_max)

        l_r_e.append(r_e)
        l_r_i.append(r_i)

    l_r_e = np.asarray(l_r_e)
    l_r_i = np.asarray(l_r_i)

    l_peak.append(np.max(l_r_e[50000:70000]))
    l_ss.append(np.mean(l_r_e[60000:70000]))

    # plotting
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

    plt.plot(l_r_e, color='blue', linewidth=plot_line_width)
    plt.plot(l_r_i, color='red', linewidth=plot_line_width)

    plt.xticks(np.arange(30000, 90000 + 5000, 20000), np.arange(0, 6 + 0.5, 2), fontsize=font_size_1, **hfont)
    plt.yticks([0, 1, 100, 10000], fontsize=font_size_1, **hfont)
    plt.xlabel('Time (s)', fontsize=font_size_1, **hfont)
    plt.ylabel('Firing rate (Hz)', fontsize=font_size_1, **hfont)
    plt.xlim([30000, 90000])
    plt.ylim([0, 10000])
    plt.legend(['Exc', 'Inh'], prop={"family": "Arial", 'size': font_size_1}, loc='upper right')
    plt.savefig('paper_figures/png/Fig_3S_Supralinear_network_2D_EI_STP_u_max_' + str(U_max) + '.png')
    plt.savefig('paper_figures/pdf/Fig_3S_Supralinear_network_2D_EI_STP_u_max_' + str(U_max) + '.pdf')


plt.figure(figsize=(figure_len, 8))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(line_width)
plt.tick_params(width=line_width, length=tick_len)

plt.yscale('symlog', linthreshy=10)

plt.plot(l_peak, color='black', linewidth=plot_line_width)
for i in range(len(l_peak)):
    plt.plot(i, l_peak[i], linestyle='none', marker='o', fillstyle='none',
             markeredgewidth=marker_edge_width, markersize=marker_size * 1.5,
             markeredgecolor='black', markerfacecolor='black')

plt.xticks([0, 1, 2], [4, 5, 6], fontsize=font_size_1, **hfont)
plt.yticks([10, 100, 1000], fontsize=font_size_1, **hfont)
plt.xlabel('U_max', fontsize=font_size_1, **hfont)
plt.ylabel('Excitatory firing rate (Hz)', fontsize=font_size_1, **hfont)
plt.xlim([-0.05, 2.05])
plt.ylim([10, 1000])
plt.legend(['Peak amplitude'], prop={"family": "Arial", 'size': font_size_1}, loc='upper right')
plt.savefig('paper_figures/png/Fig_3S_Supralinear_network_2D_EI_STP_peak_u_max.png')
plt.savefig('paper_figures/pdf/Fig_3S_Supralinear_network_2D_EI_STP_peak_u_max.pdf')

plt.figure(figsize=(figure_len, 8))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(line_width)
plt.tick_params(width=line_width, length=tick_len)

plt.plot(l_ss, color='black', linestyle='dashed', linewidth=plot_line_width)

for i in range(len(l_peak)):
    plt.plot(i, l_ss[i], linestyle='none', marker='o', fillstyle='none',
             markeredgewidth=marker_edge_width, markersize=marker_size * 1.5,
             markeredgecolor='black', markerfacecolor='black')

plt.xticks([0, 1, 2], [4, 5, 6], fontsize=font_size_1, **hfont)
plt.yticks([0, 1, 2], fontsize=font_size_1, **hfont)
plt.xlabel('U_max', fontsize=font_size_1, **hfont)
plt.ylabel('Excitatory firing rate (Hz)', fontsize=font_size_1, **hfont)
plt.xlim([-0.05, 2.05])
plt.ylim([0, 2])
plt.legend(['Steady state'], prop={"family": "Arial", 'size': font_size_1}, loc='upper right')
plt.savefig('paper_figures/png/Fig_3S_Supralinear_network_2D_EI_STP_ss_u_max.png')
plt.savefig('paper_figures/pdf/Fig_3S_Supralinear_network_2D_EI_STP_ss_u_max.pdf')