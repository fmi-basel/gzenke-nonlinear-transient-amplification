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

# input
l_g_e_temp = np.arange(1.6, 2.0+0.005, 0.01)
l_r_e_max = []

for g_e_idx in range(len(l_g_e_temp)):
    g_e_temp = l_g_e_temp[g_e_idx]

    r_e, r_i = 0, 0
    z_e, z_i = 0, 0
    l_r_e, l_r_i, l_x = [], [], []

    for i in range(T):
        if 50000 <= i < 70000:
            g_e, g_i = g_e_temp, 2
        else:
            g_e, g_i = 1.55, 2

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
        l_x.append(x)

    l_r_e_max.append(np.max(l_r_e[30000:70000])/g_e_temp)


# input
l_g_e_temp = np.arange(1.6, 2.0+0.005, 0.01)
l_r_e_max_linear = []

for g_e_idx in range(len(l_g_e_temp)):
    g_e_temp = l_g_e_temp[g_e_idx]

    r_e, r_i = 0, 0
    z_e, z_i = 0, 0
    l_r_e, l_r_i, l_x = [], [], []

    for i in range(T):
        if 50000 <= i < 70000:
            g_e, g_i = g_e_temp, 2
        else:
            g_e, g_i = 1.55, 2

        g_e = g_e * (g_e > 0)
        g_i = g_i * (g_i > 0)

        # SSN part
        z_e = Jee * x * r_e - Jei * r_i + g_e
        z_i = Jie * r_e - Jii * r_i + g_i

        z_e = z_e * (z_e > 0)
        z_i = z_i * (z_i > 0)

        r_e = r_e + (-r_e + np.power(z_e, 1)) / tau_e * dt
        r_i = r_i + (-r_i + np.power(z_i, 1)) / tau_i * dt

        r_e = r_e * (r_e > 0)
        r_i = r_i * (r_i > 0)

        # adaptation of excitatory neurons
        x = x + ((1 - x) / tau_x - u_s * x * r_e) * dt
        x = np.clip(x, 0, 1)

        l_r_e.append(r_e)
        l_r_i.append(r_i)
        l_x.append(x)

    l_r_e_max_linear.append(np.max(l_r_e[30000:70000])/g_e_temp)


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

plt.plot(l_r_e_max, linewidth=plot_line_width)
plt.plot(l_r_e_max_linear, linewidth=plot_line_width)

plt.xticks([0, 10, 20, 30, 40], [1.6, 1.7, 1.8, 1.9, 2.0], fontsize=font_size_1, **hfont)
plt.yticks([0, 1, 100, 10000], fontsize=font_size_1, **hfont)
plt.xlabel('$g_E$', fontsize=font_size_1, **hfont)
plt.ylabel('Amplification index', fontsize=font_size_1, **hfont)
plt.xlim([-0.5, len(l_g_e_temp)-0.5])
plt.ylim([0, 10000])
plt.hlines(y=1, xmin=0, xmax=40, colors='k', linestyles=[(0, (6, 6, 6, 6))], linewidth=line_width)
plt.legend(['supralinear network', 'linear network'], prop={"family": "Arial", 'size': font_size_1}, loc='upper right')
plt.savefig('paper_figures/png/Fig_3_Supralinear_linear_network_2D_EE_STP.png')
plt.savefig('paper_figures/pdf/Fig_3_Supralinear_linear_network_2D_EE_STP.pdf')