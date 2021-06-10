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

l_color = ['#D6EAF8', '#85C1E9', '#3498DB', '#2874A6']

Jee = 1.8
Jei = 1.0
Jie = 1.0
Jii = 0.6
n = 2
gI = 2

l_linestyle = ['solid', 'solid', 'solid', 'solid']

plt.figure(figsize=(figure_len, figure_width))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(line_width)
plt.tick_params(width=line_width, length=tick_len)


l_b = [0, 1, 3, 5]

plt.figure(figsize=(figure_len, figure_width))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(line_width)
plt.tick_params(width=line_width, length=tick_len)

for b_idx in range(len(l_b)):
    b = l_b[b_idx]

    gE = 3

    z = np.arange(0, 15, 0.001)

    detJ = -Jee * Jii + Jei * Jie
    C_p = -(1 / Jei) * Jii * gE + gI
    P_z = detJ * (1 / Jei) * np.power(z, 2)/(1+b) + (1 / Jei) * Jii * z + C_p
    F_z = Jee * np.power(z, 2)/(1+b) - Jei * np.power(P_z * (P_z > 0), 2) - z + gE
    plt.plot(z, F_z, color=l_color[b_idx], linestyle=l_linestyle[b_idx], linewidth=plot_line_width)

plt.hlines(y=0, xmin=0, xmax=15, colors='k', linestyles=[(0, (6, 6, 6, 6))], linewidth=line_width)
plt.xlabel(r'$z$', fontsize=font_size_1, **hfont)
plt.ylabel(r'$F(z)$', fontsize=font_size_1, **hfont)
plt.xticks([0, 5, 10, 15], fontsize=font_size_1, **hfont)
plt.yticks(np.arange(-10, 30+5, 10), fontsize=font_size_1, **hfont)
plt.xlim([0, 15])
plt.ylim([-10, 30])
plt.legend(['No adaptation', '$b:$ 1', '$b:$ 3', '$b:$ 5'], prop={"family": "Arial", 'size': font_size_1}, loc='upper right')
plt.savefig('paper_figures/png/Fig_2S_Supralinear_network_F_z_with_inputs_adaptation.png')
plt.savefig('paper_figures/pdf/Fig_2S_Supralinear_network_F_z_with_inputs_adaptation.pdf')


l_b = [200]

plt.figure(figsize=(figure_len, figure_width))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(line_width)
plt.tick_params(width=line_width, length=tick_len)

for b_idx in range(len(l_b)):
    b = l_b[b_idx]

    gE = 3

    z = np.arange(0, 2000, 0.01)

    detJ = -Jee * Jii + Jei * Jie
    C_p = -(1 / Jei) * Jii * gE + gI
    P_z = detJ * (1 / Jei) * np.power(z, 2)/(1+b) + (1 / Jei) * Jii * z + C_p
    F_z = Jee * np.power(z, 2)/(1+b) - Jei * np.power(P_z * (P_z > 0), 2) - z + gE
    plt.plot(z, F_z, color=l_color[b_idx], linestyle=l_linestyle[b_idx], linewidth=plot_line_width)

plt.hlines(y=0, xmin=0, xmax=2000, colors='k', linestyles=[(0, (6, 6, 6, 6))], linewidth=line_width)
plt.xlabel(r'$z$', fontsize=font_size_1, **hfont)
plt.ylabel(r'$F(z)$', fontsize=font_size_1, **hfont)
plt.xticks([0, 1000, 2000], fontsize=font_size_1, **hfont)
plt.yticks([-50000, -25000, 0, 25000, 50000], fontsize=font_size_1, **hfont)
plt.xlim([0, 2000])
plt.ylim([-50000, 50000])
plt.legend(['$b:$ 200'], prop={"family": "Arial", 'size': font_size_1}, loc='upper right')
plt.savefig('paper_figures/png/Fig_2S_Supralinear_network_F_z_with_inputs_strong_adaptation.png')
plt.savefig('paper_figures/pdf/Fig_2S_Supralinear_network_F_z_with_inputs_strong_adaptation.pdf')