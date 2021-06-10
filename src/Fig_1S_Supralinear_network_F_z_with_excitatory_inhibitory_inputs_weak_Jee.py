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

Jee = 0.5
Jei = 0.45
Jie = 1.0
Jii = 1.5
n = 2

l_gE = [1, 2, 3]
l_gI = [2, 3, 4]

plt.figure(figsize=(figure_len, figure_width))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(line_width)
plt.tick_params(width=line_width, length=tick_len)

for gE_idx in range(len(l_gE)):
    gE = l_gE[gE_idx]
    gI = l_gI[gE_idx]

    z = np.arange(0, 10, 0.001)

    detJ = -Jee * Jii + Jei * Jie
    C_p = -(1 / Jei) * Jii * gE + gI
    P_z = detJ * (1 / Jei) * np.power(z, 2) + (1 / Jei) * Jii * z + C_p
    F_z = Jee * np.power(z, 2) - Jei * np.power(P_z * (P_z > 0), 2) - z + gE

    plt.plot(z, F_z, color=l_color[gE_idx], linewidth=plot_line_width)

plt.hlines(y=0, xmin=0, xmax=100, colors='k', linestyles=[(0, (6, 6, 6, 6))], linewidth=line_width)
plt.xlabel(r'$z$', fontsize=font_size_1, **hfont)
plt.ylabel(r'$F(z)$', fontsize=font_size_1, **hfont)
plt.xticks(np.arange(0, 5+1, 1), fontsize=font_size_1, **hfont)
plt.yticks(np.arange(-5, 15+5, 5), fontsize=font_size_1, **hfont)
plt.xlim([0, 5])
plt.ylim([-5, 15])
plt.legend([r'$g_E, g_I$: 1, 2', r'$g_E, g_I$: 2, 3', r'$g_E, g_I$: 3, 4'], prop={"family": "Arial", 'size': font_size_1}, loc='upper right')
plt.savefig('paper_figures/png/Fig_1S_Supralinear_network_F_z_with_excitatory_inhibitory_inputs_weak_Jee.png')
plt.savefig('paper_figures/pdf/Fig_1S_Supralinear_network_F_z_with_excitatory_inhibitory_inputs_weak_Jee.pdf')

