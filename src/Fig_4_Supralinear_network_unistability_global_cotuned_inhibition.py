import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sympy.solvers import solve
from sympy import Symbol
from matplotlib import patches
import matplotlib.patches as mpatches
import scipy.io as sio
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap
from sympy.solvers import solve
from sympy import Symbol
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr

session = WolframLanguageSession()

ratio = 1.25
# plotting configuration
figure_len, figure_width = 15*1.5, 12*1.5
font_size_1, font_size_2 = 36*ratio, 36*ratio
legend_size = 18*ratio
line_width, tick_len = 3*ratio, 10*ratio
marker_size = 15*ratio
plot_line_width = 5*ratio
hfont = {'fontname': 'Arial'}

sns.set(style='ticks')

tau_e = 0.02
tau_i = 0.01

k = 0.1
l = 0.5
beta = 0.9
N = 2

e = 1/tau_e
f = 1/tau_i

l_a = np.arange(0, 201, 1)
l_d = np.arange(0, 201, 1)
uni_3D_mat = np.zeros((len(l_d), len(l_a))) * np.nan
uni_4D_mat = np.zeros((len(l_d), len(l_a))) * np.nan

example_global_unistable_x = (1.3*2*np.power(0.1619020544546218, 1/2))/tau_e
example_global_unistable_y = (0.6*2*np.power(1.0145746970782272, 1/2))/tau_i
example_global_multistable_x = (1.4*2*np.power(0.29638770180500046, 1/2))/tau_e
example_global_multistable_y = (0.6*2*np.power(1.0396121792300297, 1/2))/tau_i

for a_idx in np.arange(len(l_a)):
    a = l_a[a_idx]
    for d_idx in np.arange(len(l_d)):
        d = l_d[d_idx]
        bc = beta * a * d
        uni_3D_mat[d_idx, a_idx] = a - e - k * a

        if np.power(a - e - k * a + N * d - N * l * d + f, 2) - 4 * np.power(N, 2) * bc * np.power(1 - l, 2) > 0:
            uni_4D_mat[d_idx, a_idx] = 1 / 2 * (a - e - k * a - N * d + N * l * d - f + np.sqrt(np.power(a - e - k * a + N * d - N * l * d + f, 2) - 4 * np.power(N, 2) * bc * np.power(1 - l, 2)))
        else:
            uni_4D_mat[d_idx, a_idx] = 1 / 2 * (a - e - k * a - N * d + N * l * d - f)

plt.figure(figsize=(figure_len, figure_width))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(line_width)
plt.tick_params(width=line_width, length=tick_len)

g = sns.heatmap(uni_3D_mat, cmap='RdBu_r', vmin=-50, vmax=50)
g.set_facecolor('gray')

g.set_xticklabels(g.get_xticklabels(), rotation=0)
g.set_yticklabels(g.get_yticklabels(), rotation=0)

plt.xlabel(r"$\tau_E^{-1}J_{EE}\alpha_E r_{E}^{\frac{\alpha_E - 1}{\alpha_E}}$", fontsize=font_size_1, **hfont)
plt.ylabel(r"$\tau_I^{-1}J_{II}\alpha_I r_{I}^{\frac{\alpha_I - 1}{\alpha_I}}$", fontsize=font_size_1, **hfont)

cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=font_size_1)
cbar = ax.collections[0].colorbar
cbar.set_ticks([-25, 25])
cbar.set_ticklabels(['Uni-stable', 'Multi-stable'])

plt.xticks(np.arange(0.5, 201 + 10, 50), [0, 50, 100, 150, 200], fontsize=font_size_1, **hfont)
plt.yticks(np.arange(0.5, 201 + 10, 50), [0, 50, 100, 150, 200], fontsize=font_size_1, **hfont)
plt.xlim([0, 201])
plt.ylim([0, 201])

plt.plot(example_global_unistable_x, example_global_unistable_y, "^", markersize=marker_size*2.0, markeredgewidth=4, markeredgecolor="black")
plt.plot(example_global_multistable_x, example_global_multistable_y, "^", markersize=marker_size*2.0, markeredgewidth=4, markeredgecolor="black")

ax2 = plt.twinx()
plt.vlines(x=e/(1-k), ymin=0, ymax=2, color='black', linestyles="--", linewidth=plot_line_width)
plt.yticks([])
plt.ylim([0, 2])

plt.savefig('paper_figures/png/Fig_4_Supralinear_network_unistability_global_inhibition.png')
plt.savefig('paper_figures/pdf/Fig_4_Supralinear_network_unistability_global_inhibition.png')


plt.figure(figsize=(figure_len, figure_width))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(line_width)
plt.tick_params(width=line_width, length=tick_len)

g = sns.heatmap(uni_4D_mat, cmap='RdBu_r', vmin=-50, vmax=50)
g.set_facecolor('gray')

g.set_xticklabels(g.get_xticklabels(), rotation=0)
g.set_yticklabels(g.get_yticklabels(), rotation=0)

plt.xlabel(r"$\tau_E^{-1}J_{EE}\alpha_E r_{E}^{\frac{\alpha_E - 1}{\alpha_E}}$", fontsize=font_size_1, **hfont)
plt.ylabel(r"$\tau_I^{-1}J_{II}\alpha_I r_{I}^{\frac{\alpha_I - 1}{\alpha_I}}$", fontsize=font_size_1, **hfont)

cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=font_size_1)
cbar = ax.collections[0].colorbar
cbar.set_ticks([-25, 25])
cbar.set_ticklabels(['Uni-stable', 'Multi-stable'])

plt.xticks(np.arange(0.5, 201 + 10, 50), [0, 50, 100, 150, 200], fontsize=font_size_1, **hfont)
plt.yticks(np.arange(0.5, 201 + 10, 50), [0, 50, 100, 150, 200], fontsize=font_size_1, **hfont)
plt.xlim([0, 201])
plt.ylim([0, 201])

l_boundary = []
for a_idx in np.arange(len(l_a)):
    a = l_a[a_idx]
    l_boundary.append(((a-e-k*a) * f)/(np.power(N, 2) * beta * a * np.power(1-l, 2) - (a - e - k*a) * (N - N*l)))

l_boundary = np.asarray(l_boundary)
ax2 = plt.twinx()
sns.lineplot(data=l_boundary, color='black', linewidth=plot_line_width, ax=ax2)
plt.yticks([])
plt.ylim([0, 201])
ax2.lines[0].set_linestyle("--")

plt.savefig('paper_figures/png/Fig_4_Supralinear_network_unistability_cotuned_inhibition.png')
plt.savefig('paper_figures/pdf/Fig_4_Supralinear_network_unistability_cotuned_inhibition.png')