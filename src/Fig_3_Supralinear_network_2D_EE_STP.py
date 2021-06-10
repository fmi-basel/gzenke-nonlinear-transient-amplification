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
r_e, r_i = 0, 0
z_e, z_i = 0, 0

l_r_e, l_r_i, l_x = [], [], []

for i in range(T):
    if 50000 <= i < 70000:
        g_e, g_i = 3.0, 2
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

l_r_e = np.asarray(l_r_e)
l_r_i = np.asarray(l_r_i)

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
plt.vlines(x=40000, ymin=0, ymax=10000, colors='k', linestyles=[(0, (6, 6, 6, 6))], linewidth=line_width)
plt.vlines(x=50050, ymin=0, ymax=10000, colors='k', linestyles=[(0, (6, 6, 6, 6))], linewidth=line_width)
plt.vlines(x=60000, ymin=0, ymax=10000, colors='k', linestyles=[(0, (6, 6, 6, 6))], linewidth=line_width)
plt.vlines(x=70050, ymin=0, ymax=10000, colors='k', linestyles=[(0, (6, 6, 6, 6))], linewidth=line_width)
plt.legend(['Exc', 'Inh'], prop={"family": "Arial", 'size': font_size_1}, loc='upper right')
plt.savefig('paper_figures/png/Fig_3_Supralinear_network_2D_EE_STP_lines.png')
plt.savefig('paper_figures/pdf/Fig_3_Supralinear_network_2D_EE_STP_lines.pdf')


l_ISN_index = (Jee * alpha_e * np.power(l_r_e, (alpha_e - 1) / alpha_e) - 1) / tau_e

plt.figure(figsize=(figure_len, figure_width))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(line_width)
plt.tick_params(width=line_width, length=tick_len)
plt.yscale('symlog', linthreshy=10)

plt.plot(l_ISN_index, linewidth=plot_line_width)

plt.xticks(np.arange(30000, 90000 + 5000, 20000), np.arange(0, 6 + 0.5, 2), fontsize=font_size_1, **hfont)
plt.yticks([-100, 0, 100, 10000], fontsize=font_size_1, **hfont)
plt.xlabel('Time (s)', fontsize=font_size_1, **hfont)
plt.ylabel('ISN index', fontsize=font_size_1, **hfont)
plt.xlim([30000, 90000])
plt.ylim([-100, 10000])
plt.hlines(y=0, xmin=30000, xmax=90000, colors='k', linestyles=[(0, (6, 6, 6, 6))], linewidth=line_width)

plt.savefig('paper_figures/png/Fig_3_Supralinear_network_2D_EE_STP_ISN_index.png')
plt.savefig('paper_figures/pdf/Fig_3_Supralinear_network_2D_EE_STP_ISN_index.pdf')

# 3D plot
ratio = 1
# plotting configuration
figure_len, figure_width = 15 * 1.5, 12 * 1.5
font_size_1, font_size_2 = 36 * ratio, 36 * ratio
legend_size = 18 * ratio
line_width, tick_len = 3 * ratio, 10 * ratio
marker_size = 15 * ratio * 1.5
plot_line_width = 5 * ratio * 1.5
hfont = {'fontname': 'Arial'}

fig = plt.figure(figsize=(figure_len, figure_width))
ax = fig.add_subplot(111, projection='3d')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(line_width)
plt.tick_params(width=line_width, length=tick_len)

ax.plot(l_r_e[30000:70000], l_r_i[30000:70000], l_x[30000:70000], color='red', linewidth=plot_line_width)
ax.plot(l_r_e[70000:], l_r_i[70000:], l_x[70000:], color='blue', linewidth=plot_line_width)
ax.plot([1.257175435993853], [4.141677430796079], [2.0045870667936274], marker='o', markersize=marker_size, color='black')
ax.plot([0.04232479408727527], [1.420454723686194], [1.0419695238241007], marker='o', markersize=marker_size, color='green')

ax.tick_params(axis='both', which='major', labelsize=font_size_1)
ax.tick_params(axis='both', which='minor', labelsize=font_size_1)

plt.xticks(np.arange(0, 100 + 10, 50), fontsize=font_size_1, **hfont)
plt.yticks(np.arange(0, 400 + 10, 100), fontsize=font_size_1, **hfont)
ax.set_zticks([0, 2, 4, 6])

ax.set_xlim((0, 100))
ax.set_ylim((0, 400))
ax.set_zlim((0, 6))

plt.xlabel(r"$r_E$", fontsize=font_size_1, **hfont)
plt.ylabel(r"$r_I$", fontsize=font_size_1, **hfont)
ax.set_zlabel(r"$x$", fontsize=font_size_1, **hfont)
# ax.view_init(0, 0)
plt.savefig('paper_figures/png/Fig_3_Supralinear_network_2D_EE_STP_3D_plot.png')
plt.savefig('paper_figures/pdf/Fig_3_Supralinear_network_2D_EE_STP_3D_plot.pdf')