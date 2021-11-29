import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.patches as mpatches
import scipy.io as sio
from matplotlib.colors import LogNorm

# plotting configuration
ratio = 1.5
figure_len, figure_width = 15*ratio, 12*ratio
font_size_1, font_size_2 = 36*ratio, 36*ratio
legend_size = 18*ratio
line_width, tick_len = 3*ratio, 10*ratio
marker_size = 15*ratio
marker_edge_width = 3 * ratio
plot_line_width = 5*ratio
hfont = {'fontname': 'Arial'}
pal = sns.color_palette("deep")

# sns.set(style='ticks')

# simulation setup
dt = 0.0001
T = int(9/dt)

# neuronal parameters
tau_e, tau_i = 0.020, 0.010
alpha_e, alpha_i = 2, 2

# short-term depression
x, u_d = 1, 1
tau_x = 0.20

# network connectivity
Jee = 1.0
Jei = 0.5
Jie = 1.2
Jii = 1.0

g_e_bs = 0.5
g_i_bs = 1.0
delta_g_e = 1.8
delta_g_i = 3.0

g_e_stim = g_e_bs + delta_g_e
g_i_stim = g_i_bs + delta_g_i

r_e, r_i = 0, 0
z_e, z_i = 0, 0
x = 1

l_r_e, l_r_i, l_x = [], [], []

for i in range(T):
    if 50000 <= i < 70000:
        g_e, g_i = g_e_stim, g_i_stim
    else:
        g_e, g_i = g_e_bs, g_i_bs

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

    x = x + ((1 - x) / tau_x - u_d * x * r_e) * dt
    x = np.clip(x, 0, 1)

    l_r_e.append(r_e)
    l_r_i.append(r_i)
    l_x.append(x)

l_r_e = np.asarray(l_r_e)
l_r_i = np.asarray(l_r_i)

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

plt.plot(l_r_e, linewidth=plot_line_width)
plt.plot(l_r_i, linewidth=plot_line_width)

plt.xticks(np.arange(30000, 90000 + 5000, 20000), [0, 2, 4, 6], fontsize=font_size_1, **hfont)
plt.yticks([0, 1, 100, 10000], fontsize=font_size_1, **hfont)
plt.xlabel('Time (s)', fontsize=font_size_1, **hfont)
plt.ylabel('Firing rate (Hz)', fontsize=font_size_1, **hfont)
plt.xlim([30000, 90000])
plt.ylim([0, 10000])

plt.legend(['Exc', 'Inh'], prop={"family": "Arial", 'size': font_size_1})
plt.savefig('paper_figures/png/Revision_Fig_Point_2_11_Feedforward_inhibition_exmaple.png')
plt.savefig('paper_figures/pdf/Revision_Fig_Point_2_11_Feedforward_inhibition_exmaple.pdf')
