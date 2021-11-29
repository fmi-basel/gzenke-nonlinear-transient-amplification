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

b_plotting_examples = False
b_plotting_summary_plot = True

if b_plotting_examples:
    # simulation setup
    dt = 0.0001
    T = int(9 / dt)

    # neuronal parameters
    tau_e, tau_i = 0.020, 0.010
    l_b_SSN = [True]

    u_d = 1
    tau_y = 0.20

    # network parameters
    n_exc, n_inh = 200, 50
    n_groups = 2

    alpha_e, alpha_i = 2, 2

    J_ee = 1.5
    J_ie = 1.0
    J_ei = 1.0
    J_ii = 1.0

    J_ee_2 = J_ee * 0.1
    J_ie_2 = J_ie * 0.3
    J_ei_2 = J_ei * 0.3
    J_ii_2 = J_ii * 0.1

    b_frozen_STP = False

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

    b_save_connectivity_mat = False

    if b_save_connectivity_mat:
        # J = sio.loadmat('data/Fig_41_Pattern_separation_EI_STP.mat')['J']
        sns.set(style='ticks')
        plt.figure(figsize=(figure_len, figure_width))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.set_aspect('equal', adjustable='box')
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)

        g = sns.heatmap(J, cmap="YlGnBu", vmin=0, vmax=0.04)
        g.set_xticklabels(g.get_xticklabels(), rotation=0)
        g.set_yticklabels(g.get_yticklabels(), rotation=0)

        plt.xticks([], fontsize=font_size_1, **hfont)
        plt.yticks([], fontsize=font_size_1, **hfont)

        plt.xlabel("Neurons", fontsize=font_size_1, **hfont)
        plt.ylabel("Neurons", fontsize=font_size_1, **hfont)

        cax = plt.gcf().axes[-1]
        cax.tick_params(labelsize=font_size_1)
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([0.0, 0.02, 0.04])
        plt.savefig('paper_figures/png/Revision_2_Pattern_separation_connectivity_matrix.png')

    l_p = [1]
    p_fraction = 0.75
    g_bs = 1.5
    g_e = 3 # 4

    for p in l_p:
        r_e, r_i = np.zeros(n_exc), np.zeros(n_inh)
        z_e, z_i = np.zeros(n_exc), np.zeros(n_inh)
        l_r_e_1, l_r_e_1_2, l_r_e_2, l_r_i_1, l_r_i_2 = [], [], [], [], []
        y = np.ones((n_exc, n_exc))
        g = np.zeros(n_exc + n_inh)

        for k in range(T):
            if 50000 < k < 70000:
                g[int(n_exc / n_groups) * 0: int(n_exc / n_groups) * 0 + int(int(n_exc / n_groups) * p_fraction)] = g_e
                g[int(n_exc / n_groups) * 0 + int(int(n_exc / n_groups) * p_fraction): int(n_exc / n_groups) * 0 + int(
                    int(n_exc / n_groups))] = g_bs
                g[int(n_exc / n_groups) * 1: int(n_exc / n_groups) * 1 + int(int(n_exc / n_groups))] = g_bs
            else:
                g[: int(n_exc)] = np.ones(n_exc) * g_bs
                g[int(n_exc):] = np.ones(int(n_inh)) * 2

            g = g * (g > 0)

            # SSN part
            z_e = np.dot(J[:n_exc, :n_exc] * y, r_e) - np.dot(J[:n_exc, n_exc:], r_i) + g[:n_exc]
            z_i = np.dot(J[n_exc:, :n_exc], r_e) - np.dot(J[n_exc:, n_exc:], r_i) + g[n_exc:]

            z_e = z_e * (z_e > 0)
            z_i = z_i * (z_i > 0)

            r_e = r_e + (-r_e + np.power(z_e, alpha_e)) / tau_e * dt
            r_i = r_i + (-r_i + np.power(z_i, alpha_i)) / tau_i * dt

            # add noise
            r_e = r_e * (r_e > 0)
            r_i = r_i * (r_i > 0)

            y = y + ((1 - y) / tau_y - u_d * y * np.tile(r_e, (n_exc, 1))) * dt
            y = y * (y >= 0)
            y = y * (y <= 1)

            l_r_e_1.append(np.mean(r_e[int(n_exc / n_groups) * 0: int(int(n_exc / n_groups) * p_fraction)]))
            l_r_e_1_2.append(np.mean(r_e[int(int(n_exc / n_groups) * p_fraction): int(n_exc / n_groups) * (0 + 1)]))
            l_r_e_2.append(np.mean(r_e[int(n_exc / n_groups) * 1: int(n_exc / n_groups) * (1 + 1)]))
            l_r_i_1.append(np.mean(r_i[int(n_inh / n_groups) * 0: int(n_inh / n_groups) * (0 + 1)]))
            l_r_i_2.append(np.mean(r_i[int(n_inh / n_groups) * 1: int(n_inh / n_groups) * (1 + 1)]))

        l_r_e_1 = np.asarray(l_r_e_1)
        l_r_e_1_2 = np.asarray(l_r_e_1_2)
        l_r_e_2 = np.asarray(l_r_e_2)
        l_r_i_1 = np.asarray(l_r_i_1)
        l_r_i_2 = np.asarray(l_r_i_2)

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
        plt.plot(l_r_e_1_2, color=pal[0], linewidth=plot_line_width)
        plt.plot(l_r_e_2, color=pal[2], linewidth=plot_line_width)

        plt.xticks(np.arange(30000, 90000 + 10000, 20000), [0, 2, 4, 6], fontsize=font_size_1, **hfont)
        plt.yticks([0, 1, 100, 10000], fontsize=font_size_1, **hfont)
        plt.xlabel('Time (s)', fontsize=font_size_1, **hfont)
        plt.ylabel('Firing rate (Hz)', fontsize=font_size_1, **hfont)
        plt.xlim([30000, 90000])
        plt.ylim([0, 10000])
        plt.legend(['Exc 1 subset 2', 'Exc 2'], prop={"family": "Arial", 'size': font_size_1}, loc='upper right')

        plt.savefig('paper_figures/png/Revision_Fig_Point_1_2_Pattern_separation_activity_EE_STP_E12_E2_p_' + str(
            p) + '_g_e_' + str(g_e) + '_example.png')
        plt.savefig('paper_figures/pdf/Revision_Fig_Point_1_2_Pattern_separation_activity_EE_STP_E12_E2_p_' + str(
            p) + '_g_e_' + str(g_e) + '_example.pdf')



if b_plotting_summary_plot:
    # simulation setup
    dt = 0.0001
    T = int(9 / dt)

    # neuronal parameters
    tau_e, tau_i = 0.020, 0.010
    l_b_SSN = [True]

    u_d = 1
    tau_y = 0.20

    # network parameters
    n_exc, n_inh = 200, 50
    n_groups = 2

    alpha_e, alpha_i = 2, 2

    J_ee = 1.5
    J_ie = 1.0
    J_ei = 1.0
    J_ii = 1.0

    J_ee_2 = J_ee * 0.1
    J_ie_2 = J_ie * 0.3
    J_ei_2 = J_ei * 0.3
    J_ii_2 = J_ii * 0.1

    b_frozen_STP = False

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

    p = 1
    p_fraction = 0.75
    g_bs = 1.5

    # l_g_e = np.arange(2, 5+0.005, 0.01)
    l_g_e = np.arange(2, 5 + 0.005, 0.5)
    l_bs_ss_ratio = []

    for g_e in l_g_e:
        r_e, r_i = np.zeros(n_exc), np.zeros(n_inh)
        z_e, z_i = np.zeros(n_exc), np.zeros(n_inh)
        l_r_e_1, l_r_e_1_2, l_r_e_2, l_r_i_1, l_r_i_2 = [], [], [], [], []
        x = np.ones(int(n_exc))
        y = np.ones((n_exc, n_exc))
        g = np.zeros(n_exc + n_inh)

        for k in range(T):
            if 50000 < k < 70000:
                g[int(n_exc / n_groups) * 0: int(n_exc / n_groups) * 0 + int(int(n_exc / n_groups) * p_fraction)] = g_e
                g[int(n_exc / n_groups) * 0 + int(int(n_exc / n_groups) * p_fraction): int(n_exc / n_groups) * 0 + int(
                    int(n_exc / n_groups))] = g_bs
                g[int(n_exc / n_groups) * 1: int(n_exc / n_groups) * 1 + int(int(n_exc / n_groups))] = g_bs
            else:
                g[: int(n_exc)] = np.ones(n_exc) * g_bs
                g[int(n_exc):] = np.ones(int(n_inh)) * 2

            g = g * (g > 0)

            # SSN part
            z_e = np.dot(J[:n_exc, :n_exc] * y, r_e) - np.dot(J[:n_exc, n_exc:], r_i) + g[:n_exc]
            z_i = np.dot(J[n_exc:, :n_exc], r_e) - np.dot(J[n_exc:, n_exc:], r_i) + g[n_exc:]

            z_e = z_e * (z_e > 0)
            z_i = z_i * (z_i > 0)

            r_e = r_e + (-r_e + np.power(z_e, alpha_e)) / tau_e * dt
            r_i = r_i + (-r_i + np.power(z_i, alpha_i)) / tau_i * dt

            # add noise
            r_e = r_e * (r_e > 0)
            r_i = r_i * (r_i > 0)

            y = y + ((1 - y) / tau_y - u_d * y * np.tile(r_e, (n_exc, 1))) * dt
            y = y * (y >= 0)
            y = y * (y <= 1)

            l_r_e_1.append(np.mean(r_e[int(n_exc / n_groups) * 0: int(int(n_exc / n_groups) * p_fraction)]))
            l_r_e_1_2.append(np.mean(r_e[int(int(n_exc / n_groups) * p_fraction): int(n_exc / n_groups) * (0 + 1)]))
            l_r_e_2.append(np.mean(r_e[int(n_exc / n_groups) * 1: int(n_exc / n_groups) * (1 + 1)]))
            l_r_i_1.append(np.mean(r_i[int(n_inh / n_groups) * 0: int(n_inh / n_groups) * (0 + 1)]))
            l_r_i_2.append(np.mean(r_i[int(n_inh / n_groups) * 1: int(n_inh / n_groups) * (1 + 1)]))

        l_r_e_1 = np.asarray(l_r_e_1)
        l_r_e_1_2 = np.asarray(l_r_e_1_2)
        l_r_e_2 = np.asarray(l_r_e_2)
        l_r_i_1 = np.asarray(l_r_i_1)
        l_r_i_2 = np.asarray(l_r_i_2)

        l_bs_ss_ratio.append(np.mean(l_r_e_1_2[60000:70000]) / np.mean(l_r_e_1_2[40000:50000]))

    plt.figure(figsize=(figure_len, figure_width))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)

    plt.yscale('symlog', linthreshy=1.05)
    plt.plot(l_bs_ss_ratio, color='black', linewidth=plot_line_width)

    for i in range(len(l_bs_ss_ratio)):
        plt.plot(i, l_bs_ss_ratio[i], linestyle='none', marker='o', fillstyle='full',
                 markeredgewidth=marker_edge_width * 2, markersize=marker_size * 2,
                 markeredgecolor='black', markerfacecolor='none')

    plt.xticks([0, 2, 4, 6], [2.0, 3.0, 4.0, 5.0], fontsize=font_size_1, **hfont)
    plt.yticks([0, 1, 10], fontsize=font_size_1, **hfont)
    plt.xlabel(r'$g_E$', fontsize=font_size_1, **hfont)
    plt.ylabel('Steady state to baseline ratio', fontsize=font_size_1, **hfont)
    plt.xlim([-0.5, len(l_g_e) - 1 + 0.5])
    plt.ylim([-0.1, 10])
    plt.hlines(y=1, xmin=-0.5, xmax=len(l_g_e) - 1 + 0.5, colors='k', linestyles=[(0, (6, 6, 6, 6))],
               linewidth=line_width)
    plt.savefig(
        'paper_figures/png/Revision_Fig_Point_1_2_Pattern_separation_activity_baseline_steady_state_summary.png')
    plt.savefig(
        'paper_figures/pdf/Revision_Fig_Point_1_2_Pattern_separation_activity_baseline_steady_state_summary.pdf')