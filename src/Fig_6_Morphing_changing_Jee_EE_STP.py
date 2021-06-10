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

pal = sns.color_palette("deep")

# simulation setup
dt = 0.0001
T = int(9/dt)

# neuronal parameters
tau_e, tau_i = 0.020, 0.010
l_b_SSN = [True]

u_s = 1
tau_y = 0.20

# network parameters
n_exc, n_inh = 200, 50
n_groups = 2

alpha_e, alpha_i = 2, 2

g_bs = 1.35
g_e = 4

l_p = [0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0]

for p in l_p:

    J_ee = 0.0
    J_ie = 1.0
    J_ei = 1.0
    J_ii = 1.0

    J_ee_2 = 1.2 * 0.3
    J_ie_2 = J_ie * 0.4
    J_ei_2 = J_ei * 0.1
    J_ii_2 = J_ii * 0.1

    b_frozen_STP = False

    sns.set(style='ticks')

    J = np.zeros((n_exc + n_inh, n_exc + n_inh))

    for i in range(n_groups):
        for j in range(n_groups):
            if i == j:
                J[int(n_exc / n_groups) * i: int(n_exc / n_groups) * (i + 1),
                int(n_exc / n_groups) * i: int(n_exc / n_groups) * (i + 1)] = J_ee / (
                            int(n_exc / n_groups) - 1)  # E to E connections within the same assembly
                J[n_exc + int(n_inh / n_groups) * i: n_exc + int(n_inh / n_groups) * (i + 1),
                int(n_exc / n_groups) * i: int(n_exc / n_groups) * (i + 1)] = J_ie / (
                    int(n_exc / n_groups))  # E to I connections within the same assembly
                J[int(n_exc / n_groups) * i: int(n_exc / n_groups) * (i + 1),
                n_exc + int(n_inh / n_groups) * i: n_exc + int(n_inh / n_groups) * (i + 1)] = J_ei / (
                    int(n_inh / n_groups))  # I to E connections within the same assembly
                J[n_exc + int(n_inh / n_groups) * i: n_exc + int(n_inh / n_groups) * (i + 1),
                n_exc + int(n_inh / n_groups) * i: n_exc + int(n_inh / n_groups) * (i + 1)] = J_ii / (
                            int(n_inh / n_groups) - 1)  # I to I connections within the same assembly
            else:
                J[int(n_exc / n_groups) * i: int(n_exc / n_groups) * (i + 1),
                int(n_exc / n_groups) * j: int(n_exc / n_groups) * (j + 1)] = J_ee_2 / (
                            int(n_exc / n_groups) - 1)  # E to E connections within the same assembly
                J[n_exc + int(n_inh / n_groups) * i: n_exc + int(n_inh / n_groups) * (i + 1),
                int(n_exc / n_groups) * j: int(n_exc / n_groups) * (j + 1)] = J_ie_2 / (
                    int(n_exc / n_groups))  # E to I connections within the same assembly

                # keep I-E and I-I
                J[int(n_exc / n_groups) * i: int(n_exc / n_groups) * (i + 1),
                n_exc + int(n_inh / n_groups) * j: n_exc + int(n_inh / n_groups) * (j + 1)] = J_ei_2 / (
                    int(n_inh / n_groups))  # I to E connections across assemblies
                J[n_exc + int(n_inh / n_groups) * i: n_exc + int(n_inh / n_groups) * (i + 1),
                n_exc + int(n_inh / n_groups) * j: n_exc + int(n_inh / n_groups) * (j + 1)] = J_ii_2 / (
                    int(n_inh / n_groups))  # I to I connections across assemblies

    np.fill_diagonal(J, 0)


    r_e, r_i = np.zeros(n_exc), np.zeros(n_inh)
    z_e, z_i = np.zeros(n_exc), np.zeros(n_inh)
    l_r_e_1 , l_r_e_2, l_r_i_1, l_r_i_2 = [], [], [], []
    x = np.ones(int(n_exc))
    y = np.ones((n_exc, n_exc))
    g = np.zeros(n_exc + n_inh)

    for k in range(T):
        if 50000 < k < 70000:
            g[int(n_exc / n_groups)*0: int(n_exc / n_groups)*0+int(int(n_exc / n_groups))] = g_bs + (g_e - g_bs) * p
            g[int(n_exc / n_groups)*1: int(n_exc / n_groups)*1+int(int(n_exc / n_groups))] = g_bs + (g_e - g_bs) * (1-p)
        else:
            g[: int(n_exc)] = np.ones(n_exc) * g_bs
            g[int(n_exc):] = np.ones(int(n_inh)) * 2

        g = g * (g > 0)

        # SSN part
        z_e = np.dot(J[:n_exc, :n_exc]*y, r_e) - np.dot(J[:n_exc, n_exc:], r_i) + g[:n_exc]
        z_i = np.dot(J[n_exc:, :n_exc], r_e) - np.dot(J[n_exc:, n_exc:], r_i) + g[n_exc:]

        z_e = z_e * (z_e > 0)
        z_i = z_i * (z_i > 0)

        r_e = r_e + (-r_e + np.power(z_e, alpha_e)) / tau_e * dt
        r_i = r_i + (-r_i + np.power(z_i, alpha_i)) / tau_i * dt

        # add noise
        r_e = r_e * (r_e > 0)
        r_i = r_i * (r_i > 0)

        y = y + ((1 - y) / tau_y - u_s * y * np.tile(r_e, (n_exc, 1))) * dt
        y = y * (y >= 0)
        y = y * (y <= 1)

        l_r_e_1.append(np.mean(r_e[int(n_exc / n_groups) * 0: int(n_exc / n_groups)]))
        l_r_e_2.append(np.mean(r_e[int(n_exc / n_groups) * 1: int(n_exc / n_groups) * (1 + 1)]))
        l_r_i_1.append(np.mean(r_i[int(n_inh / n_groups) * 0: int(n_inh / n_groups) * (0 + 1)]))
        l_r_i_2.append(np.mean(r_i[int(n_inh / n_groups) * 1: int(n_inh / n_groups) * (1 + 1)]))

    l_r_e_1 = np.asarray(l_r_e_1)
    l_r_e_2 = np.asarray(l_r_e_2)
    l_r_i_1 = np.asarray(l_r_i_1)
    l_r_i_2 = np.asarray(l_r_i_2)

    sio.savemat('data/Fig_6_Morphing_activity_EE_STP_E1_Jee_' + str(J_ee) + '_p_' + str(p) + '.mat', mdict={'E1': l_r_e_1})
    sio.savemat('data/Fig_6_Morphing_activity_EE_STP_E2_Jee_' + str(J_ee) + '_p_' + str(p) + '.mat', mdict={'E2': l_r_e_2})
