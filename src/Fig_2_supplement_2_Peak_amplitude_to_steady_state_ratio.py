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
marker_size = 20*ratio
marker_edge_width = 3 * ratio
plot_line_width = 5*ratio
hfont = {'fontname': 'Arial'}
pal = sns.color_palette("deep")


b_plotting_as_a_function_of_input = True
b_plotting_as_a_function_of_Jee = False

if b_plotting_as_a_function_of_input:
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
    l_g_e_temp = np.arange(1.6, 2.4 + 0.005, 0.05)
    l_r_e_max = []
    l_pk, l_ss = [], []

    for g_e_idx in range(len(l_g_e_temp)):
        g_e_temp = l_g_e_temp[g_e_idx]

        r_e, r_i = 0, 0
        z_e, z_i = 0, 0
        l_r_e, l_r_i, l_x = [], [], []

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

        l_r_e_max.append(np.max(l_r_e[50000:70000]) / g_e_temp)
        l_pk.append(np.max(l_r_e[50000:70000]))
        l_ss.append(np.mean(l_r_e[60000:70000]))

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

    plt.plot(l_pk, '^', markersize=marker_size, markerfacecolor="None", markeredgecolor='black',
             markeredgewidth=marker_edge_width)
    plt.plot(l_ss, 'o', markersize=marker_size, markerfacecolor="None", markeredgecolor='black',
             markeredgewidth=marker_edge_width)

    plt.xticks([0, 4, 8, 12, 16], [1.6, 1.8, 2.0, 2.2, 2.4], fontsize=font_size_1, **hfont)
    plt.yticks([0, 1, 100, 10000], fontsize=font_size_1, **hfont)
    plt.xlabel('$g_E$', fontsize=font_size_1, **hfont)
    plt.ylabel('Firing rate (Hz)', fontsize=font_size_1, **hfont)
    plt.xlim([-0.5, len(l_g_e_temp) - 0.5])
    plt.ylim([0, 10000])

    plt.legend(['Peak amplitude', 'Steady state'], prop={"family": "Arial", 'size': font_size_1}, loc='upper right')
    plt.savefig('paper_figures/png/Revision_Fig_Point_2_5_NTA_2D_EE_STP_peak_ss_with_inputs.png')
    plt.savefig('paper_figures/pdf/Revision_Fig_Point_2_5_NTA_2D_EE_STP_peak_ss_with_inputs.pdf')

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

    plt.plot(np.asarray(l_pk) / np.asarray(l_ss), 'o', markersize=marker_size, color='black')

    plt.xticks([0, 4, 8, 12, 16], [1.6, 1.8, 2.0, 2.2, 2.4], fontsize=font_size_1, **hfont)
    plt.yticks([0, 1, 100, 10000], fontsize=font_size_1, **hfont)
    plt.xlabel('$g_E$', fontsize=font_size_1, **hfont)
    plt.ylabel('Peak amplitude to steady state ratio', fontsize=font_size_1, **hfont)
    plt.xlim([-0.5, len(l_g_e_temp) - 0.5])
    plt.ylim([0, 10000])
    plt.vlines(x=2, ymin=0, ymax=10000, colors='k', linestyles=[(0, (6, 6, 6, 6))], linewidth=line_width)
    plt.savefig('paper_figures/png/Revision_Fig_Point_2_5_NTA_2D_EE_STP_peak_ss_ratio_with_inputs.png')
    plt.savefig('paper_figures/pdf/Revision_Fig_Point_2_5_NTA_2D_EE_STP_peak_ss_ratio_with_inputs.pdf')


if b_plotting_as_a_function_of_Jee:
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
    l_Jee_temp = np.arange(1.0, 1.8 + 0.005, 0.05)
    l_r_e_max = []
    l_pk, l_ss = [], []

    for Jee_idx in range(len(l_Jee_temp)):
        Jee = l_Jee_temp[Jee_idx]

        r_e, r_i = 0, 0
        z_e, z_i = 0, 0
        l_r_e, l_r_i, l_x = [], [], []

        for i in range(T):
            if 50000 <= i < 70000:
                g_e, g_i = 2.0, 2
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

        l_r_e_max.append(np.max(l_r_e[50000:70000]) / 3.0)
        l_pk.append(np.max(l_r_e[50000:70000]))
        l_ss.append(np.mean(l_r_e[60000:70000]))

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

    plt.plot(l_pk, '^', markersize=marker_size, markerfacecolor="None", markeredgecolor='black',
             markeredgewidth=marker_edge_width)
    plt.plot(l_ss, 'o', markersize=marker_size, markerfacecolor="None", markeredgecolor='black',
             markeredgewidth=marker_edge_width)

    plt.xticks([0, 4, 8, 12, 16], [1.0, 1.2, 1.4, 1.6, 1.8], fontsize=font_size_1, **hfont)
    plt.yticks([0, 1, 100, 10000], fontsize=font_size_1, **hfont)
    plt.xlabel(r'$J_{EE}$', fontsize=font_size_1, **hfont)
    plt.ylabel('Firing rate (Hz)', fontsize=font_size_1, **hfont)
    plt.xlim([-0.5, len(l_Jee_temp) - 0.5])
    plt.ylim([0, 10000])
    plt.legend(['Peak amplitude', 'Steady state'], prop={"family": "Arial", 'size': font_size_1}, loc='upper right')
    plt.savefig('paper_figures/png/Revision_Fig_Point_2_5_NTA_2D_EE_STP_peak_ss_with_Jee.png')
    plt.savefig('paper_figures/pdf/Revision_Fig_Point_2_5_NTA_2D_EE_STP_peak_ss_with_Jee.pdf')

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

    plt.plot(np.asarray(l_pk) / np.asarray(l_ss), 'o', markersize=marker_size, color='black')

    plt.xticks([0, 4, 8, 12, 16], [1.0, 1.2, 1.4, 1.6, 1.8], fontsize=font_size_1, **hfont)
    plt.yticks([0, 1, 100, 10000], fontsize=font_size_1, **hfont)
    plt.xlabel(r'$J_{EE}$', fontsize=font_size_1, **hfont)
    plt.ylabel('Peak amplitude to steady state ratio', fontsize=font_size_1, **hfont)
    plt.xlim([-0.5, len(l_Jee_temp) - 0.5])
    plt.ylim([0, 10000])
    plt.vlines(x=11.2, ymin=0, ymax=10000, colors='k', linestyles=[(0, (6, 6, 6, 6))], linewidth=line_width)
    plt.savefig('paper_figures/png/Revision_Fig_Point_2_5_NTA_2D_EE_STP_peak_ss_ratio_with_Jee.png')
    plt.savefig('paper_figures/pdf/Revision_Fig_Point_2_5_NTA_2D_EE_STP_peak_ss_ratio_with_Jee.pdf')