import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sympy.solvers import solve
from sympy import Symbol
from matplotlib import patches
import matplotlib.patches as mpatches
import scipy.io as sio

# plotting
ratio = 1.5
figure_len, figure_width = 15*1.5, 12*1.5
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
a, b = 0, 200
tau_a = 0.2

# network connectivity
Jee = 1.8
Jie = 1.0
Jei = 1.0
Jii = 0.6

# input
r_e, r_i = 0, 0
z_e, z_i = 0, 0

l_r_e, l_r_i, l_a = [], [], []

for i in range(T):
    if 50000 <= i < 70000:
        g_e, g_i = 3.0, 2
    else:
        g_e, g_i = 1.55, 2

    g_e = g_e * (g_e > 0)
    g_i = g_i * (g_i > 0)

    # SSN part
    z_e = Jee * r_e - Jei * r_i + g_e
    z_i = Jie * r_e - Jii * r_i + g_i

    z_e = z_e * (z_e > 0)
    z_i = z_i * (z_i > 0)

    r_e = r_e + (-r_e + np.power(z_e, alpha_e) - a) / tau_e * dt
    r_i = r_i + (-r_i + np.power(z_i, alpha_i)) / tau_i * dt

    r_e = r_e * (r_e > 0)
    r_i = r_i * (r_i > 0)

    # adaptation of excitatory neurons
    a = a + (-a + b * r_e) * dt / tau_a

    l_r_e.append(r_e)
    l_r_i.append(r_i)
    l_a.append(a)

l_r_e = np.asarray(l_r_e)
l_r_i = np.asarray(l_r_i)
l_a = np.asarray(l_a)

print(np.max(l_r_e[60000:70000]))
print(np.min(l_r_e[60000:70000]))
print(np.max(l_r_i[60000:70000]))
print(np.min(l_r_i[60000:70000]))
print(np.max(l_a[60000:70000]))
print(np.min(l_a[60000:70000]))


# perturbed initial condition

# simulation setup
dt = 0.0001
T = int(9/dt)

# neuronal parameters
tau_e, tau_i = 0.020, 0.010
alpha_e, alpha_i = 2, 2

# adaptation
a, b = 0, 200
tau_a = 0.2

# network connectivity
Jee = 1.8
Jie = 1.0
Jei = 1.0
Jii = 0.6

# input
r_e, r_i = 0, 0
z_e, z_i = 0, 0

l_r_e_perturbed, l_r_i_perturbed, l_a_perturbed = [], [], []

for i in range(T):
    if 50000 <= i < 70000:
        g_e, g_i = 3.0, 2
    else:
        g_e, g_i = 1.55, 2

    g_e = g_e * (g_e > 0)
    g_i = g_i * (g_i > 0)

    # SSN part
    z_e = Jee * r_e - Jei * r_i + g_e
    z_i = Jie * r_e - Jii * r_i + g_i

    z_e = z_e * (z_e > 0)
    z_i = z_i * (z_i > 0)

    if i == 60000:
        r_e = 0.013277
        r_i = 1.39029
        a = 2.65541
    else:
        r_e = r_e + (-r_e + np.power(z_e, alpha_e) - a) / tau_e * dt
        r_i = r_i + (-r_i + np.power(z_i, alpha_i)) / tau_i * dt

        r_e = r_e * (r_e > 0)
        r_i = r_i * (r_i > 0)

        # adaptation of excitatory neurons
        a = a + (-a + b * r_e) * dt / tau_a

    l_r_e_perturbed.append(r_e)
    l_r_i_perturbed.append(r_i)
    l_a_perturbed.append(a)

l_r_e_perturbed = np.asarray(l_r_e_perturbed)
l_r_i_perturbed = np.asarray(l_r_i_perturbed)
l_a_perturbed = np.asarray(l_a_perturbed)



# perturbed initial condition

# simulation setup
dt = 0.0001
T = int(9/dt)

# neuronal parameters
tau_e, tau_i = 0.020, 0.010
alpha_e, alpha_i = 2, 2

# adaptation
a, b = 0, 200
tau_a = 0.2

# network connectivity
Jee = 1.8
Jie = 1.0
Jei = 1.0
Jii = 0.6

# input
r_e, r_i = 0, 0
z_e, z_i = 0, 0

l_r_e_perturbed_1, l_r_i_perturbed_1, l_a_perturbed_1 = [], [], []

for i in range(T):
    if 50000 <= i < 70000:
        g_e, g_i = 3.0, 2
    else:
        g_e, g_i = 1.55, 2

    g_e = g_e * (g_e > 0)
    g_i = g_i * (g_i > 0)

    # SSN part
    z_e = Jee * r_e - Jei * r_i + g_e
    z_i = Jie * r_e - Jii * r_i + g_i

    z_e = z_e * (z_e > 0)
    z_i = z_i * (z_i > 0)

    if i == 60000:
        # r_e = 0.013277
        # r_i = 1.39029
        # a = 2.65541

        r_e = 0.02
        r_i = 1.5
        a = 3
    else:
        r_e = r_e + (-r_e + np.power(z_e, alpha_e) - a) / tau_e * dt
        r_i = r_i + (-r_i + np.power(z_i, alpha_i)) / tau_i * dt

        r_e = r_e * (r_e > 0)
        r_i = r_i * (r_i > 0)

        # adaptation of excitatory neurons
        a = a + (-a + b * r_e) * dt / tau_a

    l_r_e_perturbed_1.append(r_e)
    l_r_i_perturbed_1.append(r_i)
    l_a_perturbed_1.append(a)

l_r_e_perturbed_1 = np.asarray(l_r_e_perturbed_1)
l_r_i_perturbed_1 = np.asarray(l_r_i_perturbed_1)
l_a_perturbed_1 = np.asarray(l_a_perturbed_1)



# perturbed initial condition

# simulation setup
dt = 0.0001
T = int(9/dt)

# neuronal parameters
tau_e, tau_i = 0.020, 0.010
alpha_e, alpha_i = 2, 2

# adaptation
a, b = 0, 200
tau_a = 0.2

# network connectivity
Jee = 1.8
Jie = 1.0
Jei = 1.0
Jii = 0.6

# input
r_e, r_i = 0, 0
z_e, z_i = 0, 0

l_r_e_perturbed_2, l_r_i_perturbed_2, l_a_perturbed_2 = [], [], []

for i in range(T):
    if 50000 <= i < 70000:
        g_e, g_i = 3.0, 2
    else:
        g_e, g_i = 1.55, 2

    g_e = g_e * (g_e > 0)
    g_i = g_i * (g_i > 0)

    # SSN part
    z_e = Jee * r_e - Jei * r_i + g_e
    z_i = Jie * r_e - Jii * r_i + g_i

    z_e = z_e * (z_e > 0)
    z_i = z_i * (z_i > 0)

    if i == 60000:
        r_e = 0.015
        r_i = 1.36
        a = 3.8
    else:
        r_e = r_e + (-r_e + np.power(z_e, alpha_e) - a) / tau_e * dt
        r_i = r_i + (-r_i + np.power(z_i, alpha_i)) / tau_i * dt

        r_e = r_e * (r_e > 0)
        r_i = r_i * (r_i > 0)

        # adaptation of excitatory neurons
        a = a + (-a + b * r_e) * dt / tau_a

    l_r_e_perturbed_2.append(r_e)
    l_r_i_perturbed_2.append(r_i)
    l_a_perturbed_2.append(a)

l_r_e_perturbed_2 = np.asarray(l_r_e_perturbed_2)
l_r_i_perturbed_2 = np.asarray(l_r_i_perturbed_2)
l_a_perturbed_2 = np.asarray(l_a_perturbed_2)




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


ax.plot(l_r_e_perturbed[60000:70000], l_r_i_perturbed[60000:70000], l_a_perturbed[60000:70000], color='black', linewidth=plot_line_width)
ax.plot(l_r_e_perturbed_1[60000:70000], l_r_i_perturbed_1[60000:70000], l_a_perturbed_1[60000:70000], color='black', linewidth=plot_line_width)
ax.plot(l_r_e_perturbed_2[60000:70000], l_r_i_perturbed_2[60000:70000], l_a_perturbed_2[60000:70000], color='black',  linewidth=plot_line_width)
ax.plot(l_r_e[60000:70000], l_r_i[60000:70000], l_a[60000:70000], color='gray', linewidth=plot_line_width)

# print(l_r_e_perturbed[60000])
# print(l_r_i_perturbed[60000])
# print(l_a_perturbed[60000])

ax.scatter(l_r_e_perturbed[60000], l_r_i_perturbed[60000], l_a_perturbed[60000], color='black', marker='o', s=marker_size*80)
ax.scatter(l_r_e_perturbed_1[60000], l_r_i_perturbed_1[60000], l_a_perturbed_1[60000], color='black', marker='o', s=marker_size*80)
ax.scatter(l_r_e_perturbed_2[60000], l_r_i_perturbed_2[60000], l_a_perturbed_2[60000], color='black', marker='o', s=marker_size*80)


ax.tick_params(axis='both', which='major', labelsize=font_size_1)
ax.tick_params(axis='both', which='minor', labelsize=font_size_1)

plt.xticks([0, 0.03, 0.06], fontsize=font_size_1, **hfont)
plt.yticks([1.35, 1.4, 1.45], fontsize=font_size_1, **hfont)
ax.set_zticks([2.5, 3.25, 4])

ax.set_xlim((0, 0.06))
ax.set_ylim((1.35, 1.45))
ax.set_zlim((2.5, 4))

plt.xlabel(r"$r_E$", fontsize=font_size_1, **hfont)
plt.ylabel(r"$r_I$", fontsize=font_size_1, **hfont)
ax.set_zlabel(r"$a$", fontsize=font_size_1, **hfont)
# ax.view_init(0, 0)
# plt.show()
plt.savefig('paper_figures/png/Revision_Fig_Supralinear_network_2D_adaptation_3D_plot.png')
plt.savefig('paper_figures/pdf/Revision_Fig_Supralinear_network_2D_adaptation_3D_plot.pdf')
