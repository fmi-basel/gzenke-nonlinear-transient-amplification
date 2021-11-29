import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
marker_edge_width = 3 * ratio
plot_line_width = 5*ratio
hfont = {'fontname': 'Arial'}
pal = sns.color_palette("deep")

b_summary_plot = True
b_example_frozen_inhibition = True
b_example_paradoxical_effect = True

if b_summary_plot:
    # simulation setup
    dt = 0.0001
    T = int(9 / dt)

    # neuronal parameters
    tau_e, tau_i = 0.020, 0.010
    alpha_e, alpha_i = 2, 2

    # short-term depression
    x, u_d = 1, 1
    tau_x = 0.20

    # network connectivity
    Jee = 1.8
    Jie = 1.0
    Jei = 1.0
    Jii = 0.6

    # input
    l_g_e_temp = np.arange(1.5, 1.8 + 0.001, 0.02)
    print(len(l_g_e_temp))
    l_ISN_index = []  # the largest eigenvalue of the Jacobian matrix
    lambda_1, lambda_2 = 0, 0

    for g_e_idx in range(len(l_g_e_temp)):
        g_e_temp = l_g_e_temp[g_e_idx]

        r_e, r_i = 0, 0
        z_e, z_i = 0, 0
        l_r_e, l_r_i, l_x = [], [], []
        Jacobian_mat = np.zeros((2, 2)) * np.nan

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

            x = x + ((1 - x) / tau_x - u_d * x * r_e) * dt
            x = np.clip(x, 0, 1)

            l_r_e.append(r_e)
            l_r_i.append(r_i)
            l_x.append(x)

        Jacobian_mat[0, 0] = 1.0 / tau_e * (
                    l_x[60000] * Jee * alpha_e * np.power(l_r_e[60000], (alpha_e - 1.0) / alpha_e) - 1)
        Jacobian_mat[0, 1] = 1.0 / tau_e * Jee * alpha_e * np.power(l_r_e[60000], (2 * alpha_e - 1.0) / alpha_e)
        Jacobian_mat[1, 0] = - u_d * l_x[60000]
        Jacobian_mat[1, 1] = -1.0 / tau_x - u_d * l_r_e[60000]
        lambda_1 = np.linalg.eig(Jacobian_mat)[0][0]
        lambda_2 = np.linalg.eig(Jacobian_mat)[0][1]
        l_ISN_index.append(np.max([lambda_1.real, lambda_2.real]))

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

    for i in range(len(l_ISN_index)):
        plt.plot(i, l_ISN_index[i], linestyle='none', marker='o', fillstyle='full', markeredgewidth=marker_edge_width * 2,
                 markersize=marker_size * 2, markeredgecolor='black', markerfacecolor='none')

    plt.xticks([0, 3, 6, 9], [3, 6, 9, 12], fontsize=font_size_1, **hfont)
    plt.plot(l_ISN_index, color='black', linewidth=plot_line_width)

    plt.xticks([0, 5, 10, 15], [1.5, 1.6, 1.7, 1.8], fontsize=font_size_1, **hfont)
    plt.yticks([-10000, -100, 0, 100, 10000], fontsize=font_size_1, **hfont)
    plt.xlabel('$g_E$', fontsize=font_size_1, **hfont)
    plt.ylabel('ISN index', fontsize=font_size_1, **hfont)
    plt.xlim([-0.5, len(l_g_e_temp) - 0.5])
    plt.ylim([-10000, 10000])
    plt.hlines(y=0, xmin=0, xmax=len(l_g_e_temp) - 1, colors='k', linestyles=[(0, (6, 6, 6, 6))], linewidth=line_width)
    plt.vlines(x=7 + 0.0044 / 0.02, ymin=-10000, ymax=10000, colors='k', linestyles=[(0, (6, 6, 6, 6))],
               linewidth=line_width)

    # ax.add_patch(patches.Rectangle((7 + 0.0044 / 0.02, 0), -10000, 10000, facecolor="grey", alpha=0.2, edgecolor="none"))

    plt.savefig('paper_figures/png/Revision_Fig_Point_2_8_ISN_index_inputs.png')
    plt.savefig('paper_figures/pdf/Revision_Fig_Point_2_8_ISN_index_inputs.pdf')


if b_example_frozen_inhibition:
    # simulation setup
    dt = 0.0001
    T = int(9 / dt)

    # neuronal parameters
    tau_e, tau_i = 0.020, 0.010
    alpha_e, alpha_i = 2, 2

    # adaptation
    x, u_d = 1, 1
    tau_x = 0.20

    # network connectivity
    Jee = 1.8
    Jie = 1.0
    Jei = 1.0
    Jii = 0.6

    x = 1
    r_e, r_i = 0, 0
    z_e, z_i = 0, 0

    l_r_e, l_r_i, l_x = [], [], []

    for i in range(T):
        g_e, g_i = 1.62, 2

        if 42000 <= i < 42001:
            r_e = r_e + 0.01
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

        if 42000 < i:
            pass
        else:
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
    l_x = np.asarray(l_x)

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
    plt.legend(['Exc', 'Inh'], prop={"family": "Arial", 'size': font_size_1}, loc='upper right')
    plt.vlines(x=42001, ymin=0, ymax=10000, colors='k', linestyles=[(0, (6, 6, 6, 6))], linewidth=line_width)

    plt.savefig(
        'paper_figures/png/Revision_Fig_Point_2_8_Supralinear_network_2D_EE_STP_normalized_activity_frozen_inhibition_ISN_no_paradoxical_effect_a.png')
    plt.savefig(
        'paper_figures/pdf/Revision_Fig_Point_2_8_Supralinear_network_2D_EE_STP_normalized_activity_frozen_inhibition_ISN_no_paradoxical_effect_a.pdf')


if b_example_paradoxical_effect:
    # simulation setup
    dt = 0.0001
    T = int(9 / dt)

    # neuronal parameters
    tau_e, tau_i = 0.020, 0.010
    alpha_e, alpha_i = 2, 2

    # short-term depression
    x, u_d = 1, 1
    tau_x = 0.20

    # network connectivity
    Jee = 1.8
    Jie = 1.0
    Jei = 1.0
    Jii = 0.6

    x = 1
    r_e, r_i = 0, 0
    z_e, z_i = 0, 0

    l_r_e, l_r_i, l_x = [], [], []

    for i in range(T):
        g_e, g_i = 1.62, 2

        if 42000 < i <= 49000:
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

        x = x + ((1 - x) / tau_x - u_d * x * r_e) * dt
        x = np.clip(x, 0, 1)

        l_r_e.append(r_e)
        l_r_i.append(r_i)
        l_x.append(x)

    l_r_e = np.asarray(l_r_e)
    l_r_i = np.asarray(l_r_i)
    l_x = np.asarray(l_x)

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
        'paper_figures/png/Revision_Fig_Point_2_8_Supralinear_network_2D_EE_STP_normalized_activity_ISN_no_paradoxical_effect_b.png')
    plt.savefig(
        'paper_figures/pdf/Revision_Fig_Point_2_8_Supralinear_network_2D_EE_STP_normalized_activity_ISN_no_paradoxical_effect_b.pdf')

