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

l_gE = [1.55, 3, 3, 1.55]
l_x = [0.9914732235747991, 0.9885924764351235, 0.6322309969432934, 0.6333349894898898]
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

for gE_idx in range(len(l_gE)):
    gE = l_gE[gE_idx]
    x = l_x[gE_idx]

    z = np.arange(0, 10, 0.001)

    detJ = -Jee * x * Jii + Jei * Jie
    C_p = -(1 / Jei) * Jii * gE + gI
    P_z = detJ * (1 / Jei) * np.power(z, 2) + (1 / Jei) * Jii * z + C_p
    F_z = Jee * x * np.power(z, 2) - Jei * np.power(P_z * (P_z > 0), 2) - z + gE
    plt.plot(z, F_z, color=l_color[gE_idx], linestyle=l_linestyle[gE_idx], linewidth=plot_line_width)

plt.hlines(y=0, xmin=0, xmax=3, colors='k', linestyles=[(0, (6, 6, 6, 6))], linewidth=line_width)
plt.xlabel(r'$z$', fontsize=font_size_1, **hfont)
plt.ylabel(r'$F(z)$', fontsize=font_size_1, **hfont)
plt.xticks([0, 1, 2, 3], fontsize=font_size_1, **hfont)
plt.yticks(np.arange(-5, 15+5, 5), fontsize=font_size_1, **hfont)
plt.xlim([0, 3])
plt.ylim([-5, 15])
plt.legend(['before stimulation', 'onset transient', 'fixed point', 'offset transient'], prop={"family": "Arial", 'size': font_size_1}, loc='upper right')
plt.savefig('paper_figures/png/Fig_3_Supralinear_network_F_z_with_inputs_EE_STD.png')
plt.savefig('paper_figures/pdf/Fig_3_Supralinear_network_F_z_with_inputs_EE_STD.pdf')