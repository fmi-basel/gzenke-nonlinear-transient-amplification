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

b_plotting_summary = False
b_plotting_example = True

if b_plotting_summary:
    # simulation setup
    dt = 0.0001
    T = int(9 / dt)

    # neuronal parameters
    tau_e, tau_i = 0.020, 0.010

    # short-term depression
    x, u_d = 1, 1
    tau_x = 0.20

    alpha_e, alpha_i = 2, 2

    # network connectivity
    l_l = [2 / 3, 1]
    k = 0.1
    l_global_e1_bs, l_global_e2_bs, l_global_e1_peak, l_global_e2_peak, l_global_e1_ss, l_global_e2_ss = [], [], [], [], [], []
    l_cotuned_e1_bs, l_cotuned_e2_bs, l_cotuned_e1_peak, l_cotuned_e2_peak, l_cotuned_e1_ss, l_cotuned_e2_ss = [], [], [], [], [], []
    l_global_amp_idx, l_cotuned_amp_idx = [], []
    l_g_e_2 = np.arange(1.5, 2.5 + 0.05, 0.1)

    for l in l_l:
        for g_e_2_temp in l_g_e_2:

            Jee = 1.6
            Jie_temp = 1.0
            Jei_temp = 1.0
            Jii_temp = 1.2
            Jie = (2 - l) * Jie_temp
            Jei = (2 - l) * Jei_temp
            Jii = (2 - l) * Jii_temp

            Jee_2 = k * Jee
            Jie_2 = l * Jie_temp
            Jei_2 = l * Jei_temp
            Jii_2 = l * Jii_temp

            r_e_1, r_e_2, r_i_1, r_i_2 = 0, 0, 0, 0
            z_e_1, z_e_2, z_i_1, z_i_2 = 0, 0, 0, 0
            x_1_1, x_1_2, x_2_1, x_2_2 = 1, 1, 1, 1
            l_r_e_1, l_r_e_2, l_r_i_1, l_r_i_2 = [], [], [], []

            for i in range(T):
                if 50000 <= i < 70000:
                    g_e_1, g_e_2, g_i_1, g_i_2 = 1.5, g_e_2_temp, 2.5, 2.5
                else:
                    g_e_1, g_e_2, g_i_1, g_i_2 = 1.5, 1.5, 2.5, 2.5

                g_e_1 = g_e_1 * (g_e_1 > 0)
                g_e_2 = g_e_2 * (g_e_2 > 0)
                g_i_1 = g_i_1 * (g_i_1 > 0)
                g_i_2 = g_i_2 * (g_i_2 > 0)

                # SSN part
                z_e_1 = Jee * x_1_1 * r_e_1 + Jee_2 * x_1_2 * r_e_2 - Jei * r_i_1 - Jei_2 * r_i_2 + g_e_1
                z_e_2 = Jee_2 * x_2_1 * r_e_1 + Jee * x_2_2 * r_e_2 - Jei_2 * r_i_1 - Jei * r_i_2 + g_e_2
                z_i_1 = Jie * r_e_1 + Jie_2 * r_e_2 - Jii * r_i_1 - Jii_2 * r_i_2 + g_i_1
                z_i_2 = Jie_2 * r_e_1 + Jie * r_e_2 - Jii_2 * r_i_1 - Jii * r_i_2 + g_i_2

                z_e_1 = z_e_1 * (z_e_1 > 0)
                z_e_2 = z_e_2 * (z_e_2 > 0)
                z_i_1 = z_i_1 * (z_i_1 > 0)
                z_i_2 = z_i_2 * (z_i_2 > 0)

                r_e_1 = r_e_1 + (-r_e_1 + np.power(z_e_1, alpha_e)) / tau_e * dt
                r_e_2 = r_e_2 + (-r_e_2 + np.power(z_e_2, alpha_e)) / tau_e * dt
                r_i_1 = r_i_1 + (-r_i_1 + np.power(z_i_1, alpha_i)) / tau_i * dt
                r_i_2 = r_i_2 + (-r_i_2 + np.power(z_i_2, alpha_i)) / tau_i * dt

                r_e_1 = r_e_1 * (r_e_1 > 0)
                r_e_2 = r_e_2 * (r_e_2 > 0)
                r_i_1 = r_i_1 * (r_i_1 > 0)
                r_i_2 = r_i_2 * (r_i_2 > 0)

                x_1_1 = x_1_1 + ((1 - x_1_1) / tau_x - u_d * x_1_1 * r_e_1) * dt
                x_1_1 = np.clip(x_1_1, 0, 1)
                x_1_2 = x_1_2 + ((1 - x_1_2) / tau_x - u_d * x_1_2 * r_e_2) * dt
                x_1_2 = np.clip(x_1_2, 0, 1)
                x_2_1 = x_2_1 + ((1 - x_2_1) / tau_x - u_d * x_2_1 * r_e_1) * dt
                x_2_1 = np.clip(x_2_1, 0, 1)
                x_2_2 = x_2_2 + ((1 - x_2_2) / tau_x - u_d * x_2_2 * r_e_2) * dt
                x_2_2 = np.clip(x_2_2, 0, 1)

                l_r_e_1.append(r_e_1)
                l_r_e_2.append(r_e_2)
                l_r_i_1.append(r_i_1)
                l_r_i_2.append(r_i_2)

            l_r_e_1 = np.asarray(l_r_e_1)
            l_r_e_2 = np.asarray(l_r_e_2)
            l_r_i_1 = np.asarray(l_r_i_1)
            l_r_i_2 = np.asarray(l_r_i_2)

            if l == 1:
                l_global_e1_bs.append(np.mean(l_r_e_1[40000:50000]))
                l_global_e2_bs.append(np.mean(l_r_e_2[40000:50000]))
                l_global_e1_peak.append(np.max(l_r_e_1[50000:60000]))
                l_global_e2_peak.append(np.max(l_r_e_2[50000:60000]))
                l_global_e1_ss.append(np.mean(l_r_e_1[60000:70000]))
                l_global_e2_ss.append(np.mean(l_r_e_2[60000:70000]))
                l_global_amp_idx.append(np.max(l_r_e_2[50000:60000]) / g_e_2_temp)
            else:
                l_cotuned_e1_bs.append(np.mean(l_r_e_1[40000:50000]))
                l_cotuned_e2_bs.append(np.mean(l_r_e_2[40000:50000]))
                l_cotuned_e1_peak.append(np.max(l_r_e_1[50000:60000]))
                l_cotuned_e2_peak.append(np.max(l_r_e_2[50000:60000]))
                l_cotuned_e1_ss.append(np.mean(l_r_e_1[60000:70000]))
                l_cotuned_e2_ss.append(np.mean(l_r_e_2[60000:70000]))
                l_cotuned_amp_idx.append(np.max(l_r_e_2[50000:60000]) / g_e_2_temp)

    # plotting configuration
    ratio = 1.5
    figure_len, figure_width = 15 * ratio, 12 * ratio
    font_size_1, font_size_2 = 36 * ratio, 36 * ratio
    legend_size = 18 * ratio
    line_width, tick_len = 3 * ratio, 10 * ratio
    marker_size = 15 * ratio
    marker_edge_width = 3 * ratio
    plot_line_width = 5 * ratio
    hfont = {'fontname': 'Arial'}
    pal = sns.color_palette("deep")

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

    plt.plot(l_global_amp_idx, linewidth=plot_line_width)
    plt.plot(l_cotuned_amp_idx, linewidth=plot_line_width)
    plt.plot(l_global_amp_idx, linestyle='none', marker='o', fillstyle='full', markeredgewidth=marker_edge_width * 2,
             markersize=marker_size * 2, markeredgecolor='black', markerfacecolor='none')
    plt.plot(l_cotuned_amp_idx, linestyle='none', marker='o', fillstyle='full', markeredgewidth=marker_edge_width * 2,
             markersize=marker_size * 2, markeredgecolor='black', markerfacecolor='none')
    plt.hlines(y=1, xmin=-0.5, xmax=len(l_g_e_2) -1 + 0.5, colors='k', linestyles=[(0, (6, 6, 6, 6))],
               linewidth=line_width)

    plt.xticks([0, 5, 10], [1.5, 2.0, 2.5], fontsize=font_size_1, **hfont)
    plt.yticks([0, 1, 100, 10000], fontsize=font_size_1, **hfont)
    plt.xlabel('$g_E$', fontsize=font_size_1, **hfont)
    plt.ylabel('Amplification index', fontsize=font_size_1, **hfont)
    plt.xlim([-0.5, len(l_g_e_2) -1 + 0.5])
    plt.ylim([-0.02, 10000])
    plt.legend(['global inhibition', 'co-tuned inhibition'], prop={"family": "Arial", 'size': font_size_1})

    plt.savefig('paper_figures/png/Revision_Fig_Point_2_6_global_cotuned_inhibition_NTA.png')
    plt.savefig('paper_figures/pdf/Revision_Fig_Point_2_6_global_cotuned_inhibition_NTA.pdf')



if b_plotting_example:
    # simulation setup
    dt = 0.0001
    T = int(9 / dt)

    # neuronal parameters
    tau_e, tau_i = 0.020, 0.010

    # short-term depression
    x, u_d = 1, 1
    tau_x = 0.20

    alpha_e, alpha_i = 2, 2

    # network connectivity
    l_l = [2 / 3, 1]
    k = 0.1
    l_global_e1_bs, l_global_e2_bs, l_global_e1_peak, l_global_e2_peak, l_global_e1_ss, l_global_e2_ss = [], [], [], [], [], []
    l_cotuned_e1_bs, l_cotuned_e2_bs, l_cotuned_e1_peak, l_cotuned_e2_peak, l_cotuned_e1_ss, l_cotuned_e2_ss = [], [], [], [], [], []
    l_global_amp_idx, l_cotuned_amp_idx = [], []
    l_g_e_2 = [2.5]

    for l in l_l:
        for g_e_2_temp in l_g_e_2:

            Jee = 1.6
            Jie_temp = 1.0
            Jei_temp = 1.0
            Jii_temp = 1.2
            Jie = (2 - l) * Jie_temp
            Jei = (2 - l) * Jei_temp
            Jii = (2 - l) * Jii_temp

            Jee_2 = k * Jee
            Jie_2 = l * Jie_temp
            Jei_2 = l * Jei_temp
            Jii_2 = l * Jii_temp

            r_e_1, r_e_2, r_i_1, r_i_2 = 0, 0, 0, 0
            z_e_1, z_e_2, z_i_1, z_i_2 = 0, 0, 0, 0
            x_1_1, x_1_2, x_2_1, x_2_2 = 1, 1, 1, 1
            l_r_e_1, l_r_e_2, l_r_i_1, l_r_i_2 = [], [], [], []

            for i in range(T):
                if 50000 <= i < 70000:
                    g_e_1, g_e_2, g_i_1, g_i_2 = 1.5, g_e_2_temp, 2.5, 2.5
                else:
                    g_e_1, g_e_2, g_i_1, g_i_2 = 1.5, 1.5, 2.5, 2.5

                g_e_1 = g_e_1 * (g_e_1 > 0)
                g_e_2 = g_e_2 * (g_e_2 > 0)
                g_i_1 = g_i_1 * (g_i_1 > 0)
                g_i_2 = g_i_2 * (g_i_2 > 0)

                # SSN part
                z_e_1 = Jee * x_1_1 * r_e_1 + Jee_2 * x_1_2 * r_e_2 - Jei * r_i_1 - Jei_2 * r_i_2 + g_e_1
                z_e_2 = Jee_2 * x_2_1 * r_e_1 + Jee * x_2_2 * r_e_2 - Jei_2 * r_i_1 - Jei * r_i_2 + g_e_2
                z_i_1 = Jie * r_e_1 + Jie_2 * r_e_2 - Jii * r_i_1 - Jii_2 * r_i_2 + g_i_1
                z_i_2 = Jie_2 * r_e_1 + Jie * r_e_2 - Jii_2 * r_i_1 - Jii * r_i_2 + g_i_2

                z_e_1 = z_e_1 * (z_e_1 > 0)
                z_e_2 = z_e_2 * (z_e_2 > 0)
                z_i_1 = z_i_1 * (z_i_1 > 0)
                z_i_2 = z_i_2 * (z_i_2 > 0)

                r_e_1 = r_e_1 + (-r_e_1 + np.power(z_e_1, alpha_e)) / tau_e * dt
                r_e_2 = r_e_2 + (-r_e_2 + np.power(z_e_2, alpha_e)) / tau_e * dt
                r_i_1 = r_i_1 + (-r_i_1 + np.power(z_i_1, alpha_i)) / tau_i * dt
                r_i_2 = r_i_2 + (-r_i_2 + np.power(z_i_2, alpha_i)) / tau_i * dt

                r_e_1 = r_e_1 * (r_e_1 > 0)
                r_e_2 = r_e_2 * (r_e_2 > 0)
                r_i_1 = r_i_1 * (r_i_1 > 0)
                r_i_2 = r_i_2 * (r_i_2 > 0)

                x_1_1 = x_1_1 + ((1 - x_1_1) / tau_x - u_d * x_1_1 * r_e_1) * dt
                x_1_1 = np.clip(x_1_1, 0, 1)
                x_1_2 = x_1_2 + ((1 - x_1_2) / tau_x - u_d * x_1_2 * r_e_2) * dt
                x_1_2 = np.clip(x_1_2, 0, 1)
                x_2_1 = x_2_1 + ((1 - x_2_1) / tau_x - u_d * x_2_1 * r_e_1) * dt
                x_2_1 = np.clip(x_2_1, 0, 1)
                x_2_2 = x_2_2 + ((1 - x_2_2) / tau_x - u_d * x_2_2 * r_e_2) * dt
                x_2_2 = np.clip(x_2_2, 0, 1)

                l_r_e_1.append(r_e_1)
                l_r_e_2.append(r_e_2)
                l_r_i_1.append(r_i_1)
                l_r_i_2.append(r_i_2)

            l_r_e_1 = np.asarray(l_r_e_1)
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
            plt.yscale('symlog', linthreshy=0.001)

            plt.plot(l_r_e_1, linewidth=plot_line_width)
            plt.plot(l_r_e_2, linewidth=plot_line_width)
            plt.plot(l_r_i_1, linewidth=plot_line_width)
            plt.plot(l_r_i_2, linewidth=plot_line_width)

            plt.xticks(np.arange(30000, 90000 + 5000, 20000), [0, 2, 4, 6], fontsize=font_size_1, **hfont)
            plt.yticks([0, 0.01, 1, 100, 10000], fontsize=font_size_1, **hfont)
            plt.xlabel('Time (s)', fontsize=font_size_1, **hfont)
            plt.ylabel('Firing rate (Hz)', fontsize=font_size_1, **hfont)
            plt.xlim([30000, 90000])
            plt.ylim([0, 10000])

            plt.legend(['Exc 1', 'Exc 2', 'Inh 1', 'Inh 2'], prop={"family": "Arial", 'size': font_size_1})
            if l == 1:
                plt.savefig('paper_figures/png/Revision_Fig_Point_2_6_global_cotuned_inhibition_NTA_global_example.png')
                plt.savefig('paper_figures/pdf/Revision_Fig_Point_2_6_global_cotuned_inhibition_NTA_global_example.pdf')
            else:
                plt.savefig(
                    'paper_figures/png/Revision_Fig_Point_2_6_global_cotuned_inhibition_NTA_cotuned_example.png')
                plt.savefig(
                    'paper_figures/pdf/Revision_Fig_Point_2_6_global_cotuned_inhibition_NTA_cotuned_example.pdf')

