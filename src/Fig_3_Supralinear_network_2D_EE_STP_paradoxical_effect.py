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
hfont = {'fontname': 'Arial'}

# simulation setup
dt = 0.0001
T = int(9/dt)

# neuronal parameters
tau_e, tau_i = 0.020, 0.010
alpha_e, alpha_i = 2, 2

# adaptation
x, u_s = 1, 1
tau_x = 0.20

# network connectivity
Jee = 1.8
Jie = 1.0
Jei = 1.0
Jii = 0.6

l_b_before_stimulation = [True, False]

for b_before_stimulation in l_b_before_stimulation:
    x = 1
    r_e, r_i = 0, 0
    z_e, z_i = 0, 0

    l_r_e, l_r_i = [], []

    for i in range(T):
        if 50000 <= i < 70000:
            g_e, g_i = 3.0, 2
        else:
            g_e, g_i = 1.55, 2

        if b_before_stimulation:
            if 42000 < i <= 49000:
                g_i = 2.1
            else:
                pass
        else:
            if 62000 < i <= 69000:
                g_i = 2.1
            else:
                pass

        g_e = g_e * (g_e > 0)
        g_i = g_i * (g_i > 0)

        # SSN part
        z_e = Jee * x * r_e - Jei * r_i + g_e
        z_i = Jie * r_e - Jii * r_i + g_i

        z_e = z_e * (z_e > 0)
        z_i = z_i * (z_i > 0)

        r_e = r_e + (-r_e + np.power(z_e, alpha_e)) / tau_e * dt
        r_i = r_i + (-r_i + np.power(z_i, alpha_i)) / tau_i * dt

        r_e = r_e * (r_e > 0)
        r_i = r_i * (r_i > 0)

        # adaptation of excitatory neurons
        x = x + ((1 - x) / tau_x - u_s * x * r_e) * dt
        x = np.clip(x, 0, 1)

        l_r_e.append(r_e)
        l_r_i.append(r_i)

    l_r_e = np.asarray(l_r_e)
    l_r_i = np.asarray(l_r_i)

    if b_before_stimulation:
        plt.figure(figsize=(figure_len, figure_width))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)

        mean_e = l_r_e / np.mean(l_r_e[40000:42000])
        mean_i = l_r_i / np.mean(l_r_i[40000:42000])

        plt.plot(mean_e, color='blue', linewidth=plot_line_width)
        plt.plot(mean_i, color='red', linewidth=plot_line_width)

        plt.xticks([40000, 42000, 44000, 46000, 48000], [1.0, 1.2, 1.4, 1.6, 1.8], fontsize=font_size_1, **hfont)
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2], fontsize=font_size_1, **hfont)
        plt.xlabel('Time (s)', fontsize=font_size_1, **hfont)
        plt.ylabel('Normalized firing rate', fontsize=font_size_1, **hfont)
        plt.xlim([40000, 48000])
        plt.ylim([0, 1.2])
        plt.legend(['Exc', 'Inh'], prop={"family": "Arial", 'size': font_size_1})
        plt.hlines(y=1, xmin=42000, xmax=50000, colors='k', linestyles=[(0, (6, 6, 6, 6))], linewidth=line_width)
        plt.savefig(
            'paper_figures/png/Fig_3_Supralinear_network_2D_EE_STP_normalized_activity_paradoxical_effect_before_stimulation.png')
        plt.savefig(
            'paper_figures/pdf/Fig_3_Supralinear_network_2D_EE_STP_normalized_activity_paradoxical_effect_before_stimulation.pdf')

    else:
        plt.figure(figsize=(figure_len, figure_width))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)

        mean_e = l_r_e / np.mean(l_r_e[60000:62000])
        mean_i = l_r_i / np.mean(l_r_i[60000:62000])

        plt.plot(mean_e, color='blue', linewidth=plot_line_width)
        plt.plot(mean_i, color='red', linewidth=plot_line_width)

        plt.xticks([60000, 62000, 64000, 66000, 68000], [3.0, 3.2, 3.4, 3.6, 3.8], fontsize=font_size_1,
                   **hfont)
        plt.yticks([0.85, 0.9, 0.95, 1.0, 1.05], fontsize=font_size_1, **hfont)
        plt.xlabel('Time (s)', fontsize=font_size_1, **hfont)
        plt.ylabel('Normalized firing rate', fontsize=font_size_1, **hfont)
        plt.xlim([60000, 68000])
        plt.ylim([0.85, 1.05])
        plt.legend(['Exc', 'Inh'], prop={"family": "Arial", 'size': font_size_1})
        plt.hlines(y=1, xmin=62000, xmax=70000, colors='k', linestyles=[(0, (6, 6, 6, 6))], linewidth=line_width)
        plt.savefig(
            'paper_figures/png/Fig_3_Supralinear_network_2D_EE_STP_normalized_activity_paradoxical_effect_during_stimulation.png')
        plt.savefig(
            'paper_figures/pdf/Fig_3_Supralinear_network_2D_EE_STP_normalized_activity_paradoxical_effect_during_stimulation.pdf')
