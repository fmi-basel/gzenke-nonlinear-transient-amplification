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

ratio_80, ratio_85, ratio_90, ratio_95, ratio_100, ratio_105, ratio_110, ratio_115, ratio_120, ratio_125, ratio_130, ratio_135, ratio_140 = [], [], [], [], [], [], [], [], [], [], [], [], []
n_loop = 20
for loop_idx in range(n_loop):
    bs_80 = sio.loadmat(
        'data_sum/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_mean_bl_firing_amp_80_' + str(
            loop_idx) + '.mat')['mean_bl_firing_4_2'][0]
    ss_80 = sio.loadmat(
        'data_sum/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_mean_ss_firing_amp_80_' + str(
            loop_idx) + '.mat')['mean_ss_firing_4_2'][0]

    bs_85 = sio.loadmat(
        'data_sum/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_mean_bl_firing_amp_85_' + str(
            loop_idx) + '.mat')['mean_bl_firing_4_2'][0]
    ss_85 = sio.loadmat(
        'data_sum/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_mean_ss_firing_amp_85_' + str(
            loop_idx) + '.mat')['mean_ss_firing_4_2'][0]

    bs_90 = sio.loadmat(
        'data_sum/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_mean_bl_firing_amp_90_' + str(
            loop_idx) + '.mat')['mean_bl_firing_4_2'][0]
    ss_90 = sio.loadmat(
        'data_sum/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_mean_ss_firing_amp_90_' + str(
            loop_idx) + '.mat')['mean_ss_firing_4_2'][0]

    bs_95 = sio.loadmat(
        'data_sum/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_mean_bl_firing_amp_95_' + str(
            loop_idx) + '.mat')['mean_bl_firing_4_2'][0]
    ss_95 = sio.loadmat(
        'data_sum/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_mean_ss_firing_amp_95_' + str(
            loop_idx) + '.mat')['mean_ss_firing_4_2'][0]

    bs_100 = sio.loadmat(
        'data_sum/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_mean_bl_firing_amp_100_' + str(
            loop_idx) + '.mat')['mean_bl_firing_4_2'][0]
    ss_100 = sio.loadmat(
        'data_sum/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_mean_ss_firing_amp_100_' + str(
            loop_idx) + '.mat')['mean_ss_firing_4_2'][0]

    bs_105 = sio.loadmat(
        'data_sum/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_mean_bl_firing_amp_105_' + str(
            loop_idx) + '.mat')['mean_bl_firing_4_2'][0]
    ss_105 = sio.loadmat(
        'data_sum/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_mean_ss_firing_amp_105_' + str(
            loop_idx) + '.mat')['mean_ss_firing_4_2'][0]

    bs_110 = sio.loadmat(
        'data_sum/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_mean_bl_firing_amp_110_' + str(
            loop_idx) + '.mat')['mean_bl_firing_4_2'][0]
    ss_110 = sio.loadmat(
        'data_sum/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_mean_ss_firing_amp_110_' + str(
            loop_idx) + '.mat')['mean_ss_firing_4_2'][0]

    bs_115 = sio.loadmat(
        'data_sum/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_mean_bl_firing_amp_115_' + str(
            loop_idx) + '.mat')['mean_bl_firing_4_2'][0]
    ss_115 = sio.loadmat(
        'data_sum/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_mean_ss_firing_amp_115_' + str(
            loop_idx) + '.mat')['mean_ss_firing_4_2'][0]

    bs_120 = sio.loadmat(
        'data_sum/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_mean_bl_firing_amp_120_' + str(
            loop_idx) + '.mat')['mean_bl_firing_4_2'][0]
    ss_120 = sio.loadmat(
        'data_sum/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_mean_ss_firing_amp_120_' + str(
            loop_idx) + '.mat')['mean_ss_firing_4_2'][0]

    bs_125 = sio.loadmat(
        'data_sum/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_mean_bl_firing_amp_125_' + str(
            loop_idx) + '.mat')['mean_bl_firing_4_2'][0]
    ss_125 = sio.loadmat(
        'data_sum/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_mean_ss_firing_amp_125_' + str(
            loop_idx) + '.mat')['mean_ss_firing_4_2'][0]

    bs_130 = sio.loadmat(
        'data_sum/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_mean_bl_firing_amp_130_' + str(
            loop_idx) + '.mat')['mean_bl_firing_4_2'][0]
    ss_130 = sio.loadmat(
        'data_sum/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_mean_ss_firing_amp_130_' + str(
            loop_idx) + '.mat')['mean_ss_firing_4_2'][0]

    bs_135 = sio.loadmat(
        'data_sum/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_mean_bl_firing_amp_135_' + str(
            loop_idx) + '.mat')['mean_bl_firing_4_2'][0]
    ss_135 = sio.loadmat(
        'data_sum/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_mean_ss_firing_amp_135_' + str(
            loop_idx) + '.mat')['mean_ss_firing_4_2'][0]

    bs_140 = sio.loadmat(
        'data_sum/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_mean_bl_firing_amp_140_' + str(
            loop_idx) + '.mat')['mean_bl_firing_4_2'][0]
    ss_140 = sio.loadmat(
        'data_sum/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_mean_ss_firing_amp_140_' + str(
            loop_idx) + '.mat')['mean_ss_firing_4_2'][0]

    ratio_80.append(ss_80 / bs_80)
    ratio_85.append(ss_85 / bs_85)
    ratio_90.append(ss_90 / bs_90)
    ratio_95.append(ss_95 / bs_95)
    ratio_100.append(ss_100 / bs_100)
    ratio_105.append(ss_105 / bs_105)
    ratio_110.append(ss_110 / bs_110)
    ratio_115.append(ss_115 / bs_115)
    ratio_120.append(ss_120 / bs_120)
    ratio_125.append(ss_125 / bs_125)
    ratio_130.append(ss_130 / bs_130)
    ratio_135.append(ss_135 / bs_135)
    ratio_140.append(ss_140 / bs_140)

# plotting
plt.figure(figsize=(figure_len, figure_width))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(line_width)
plt.tick_params(width=line_width, length=tick_len)
# plt.yscale('symlog', linthreshy=1)

# sns.boxplot(data=[ratio_80, ratio_85, ratio_90, ratio_95, ratio_100, ratio_105, ratio_110, ratio_115, ratio_120, ratio_125, ratio_130, ratio_135, ratio_140], width=0.4, linewidth=line_width)
ax = sns.boxplot(data=[ratio_80, ratio_90, ratio_100, ratio_110, ratio_120, ratio_130, ratio_140], width=0.45,
                 linewidth=line_width, color='white')  # , showfliers = False)

print(len(ax.lines))
# iterate over boxes
for m, box in enumerate(ax.artists):
    print(m)
    box.set_edgecolor('black')
    box.set_facecolor('white')

    # iterate over whiskers and median lines
    for j in range(6 * m, 6 * (m + 1)):
        #         print(j)
        ax.lines[j].set_color('black')

# plot the data points
for i in range(len(ratio_80)):
    if i % 2 == 0:
        plt.plot(0 - 0.1, ratio_80[i], linestyle='none', marker='o', fillstyle='full',
                 markeredgewidth=marker_edge_width, markersize=marker_size,
                 markeredgecolor='black', markerfacecolor='none')
    else:
        plt.plot(0 + 0.1, ratio_80[i], linestyle='none', marker='o', fillstyle='full',
                 markeredgewidth=marker_edge_width, markersize=marker_size,
                 markeredgecolor='black', markerfacecolor='none')

# for i in range(len(ratio_85)):
#     if i%2 == 0:
#         plt.plot(1 - 0.1, ratio_85[i], linestyle='none', marker='o', fillstyle='full',
#                  markeredgewidth=marker_edge_width, markersize=marker_size,
#                  markeredgecolor='black', markerfacecolor='none')
#     else:
#         plt.plot(1 + 0.1, ratio_85[i], linestyle='none', marker='o', fillstyle='full',
#                  markeredgewidth=marker_edge_width, markersize=marker_size,
#                  markeredgecolor='black', markerfacecolor='none')

for i in range(len(ratio_90)):
    if i % 2 == 0:
        plt.plot(1 - 0.1, ratio_90[i], linestyle='none', marker='o', fillstyle='full',
                 markeredgewidth=marker_edge_width, markersize=marker_size,
                 markeredgecolor='black', markerfacecolor='none')
    else:
        plt.plot(1 + 0.1, ratio_90[i], linestyle='none', marker='o', fillstyle='full',
                 markeredgewidth=marker_edge_width, markersize=marker_size,
                 markeredgecolor='black', markerfacecolor='none')

# for i in range(len(ratio_95)):
#     if i%2 == 0:
#         plt.plot(3 - 0.1, ratio_95[i], linestyle='none', marker='o', fillstyle='full',
#                  markeredgewidth=marker_edge_width, markersize=marker_size,
#                  markeredgecolor='black', markerfacecolor='none')
#     else:
#         plt.plot(3 + 0.1, ratio_95[i], linestyle='none', marker='o', fillstyle='full',
#                  markeredgewidth=marker_edge_width, markersize=marker_size,
#                  markeredgecolor='black', markerfacecolor='none')


for i in range(len(ratio_100)):
    if i % 2 == 0:
        plt.plot(2 - 0.1, ratio_100[i], linestyle='none', marker='o', fillstyle='full',
                 markeredgewidth=marker_edge_width, markersize=marker_size,
                 markeredgecolor='black', markerfacecolor='none')
    else:
        plt.plot(2 + 0.1, ratio_100[i], linestyle='none', marker='o', fillstyle='full',
                 markeredgewidth=marker_edge_width, markersize=marker_size,
                 markeredgecolor='black', markerfacecolor='none')

# for i in range(len(ratio_105)):
#     if i%2 == 0:
#         plt.plot(5 - 0.1, ratio_105[i], linestyle='none', marker='o', fillstyle='full',
#                  markeredgewidth=marker_edge_width, markersize=marker_size,
#                  markeredgecolor='black', markerfacecolor='none')
#     else:
#         plt.plot(5 + 0.1, ratio_105[i], linestyle='none', marker='o', fillstyle='full',
#                  markeredgewidth=marker_edge_width, markersize=marker_size,
#                  markeredgecolor='black', markerfacecolor='none')

for i in range(len(ratio_110)):
    if i % 2 == 0:
        plt.plot(3 - 0.1, ratio_110[i], linestyle='none', marker='o', fillstyle='full',
                 markeredgewidth=marker_edge_width, markersize=marker_size,
                 markeredgecolor='black', markerfacecolor='none')
    else:
        plt.plot(3 + 0.1, ratio_110[i], linestyle='none', marker='o', fillstyle='full',
                 markeredgewidth=marker_edge_width, markersize=marker_size,
                 markeredgecolor='black', markerfacecolor='none')

# for i in range(len(ratio_115)):
#     if i%2 == 0:
#         plt.plot(7 - 0.1, ratio_115[i], linestyle='none', marker='o', fillstyle='full',
#                  markeredgewidth=marker_edge_width, markersize=marker_size,
#                  markeredgecolor='black', markerfacecolor='none')
#     else:
#         plt.plot(7 + 0.1, ratio_115[i], linestyle='none', marker='o', fillstyle='full',
#                  markeredgewidth=marker_edge_width, markersize=marker_size,
#                  markeredgecolor='black', markerfacecolor='none')

for i in range(len(ratio_120)):
    if i % 2 == 0:
        plt.plot(4 - 0.1, ratio_120[i], linestyle='none', marker='o', fillstyle='full',
                 markeredgewidth=marker_edge_width, markersize=marker_size,
                 markeredgecolor='black', markerfacecolor='none')
    else:
        plt.plot(4 + 0.1, ratio_120[i], linestyle='none', marker='o', fillstyle='full',
                 markeredgewidth=marker_edge_width, markersize=marker_size,
                 markeredgecolor='black', markerfacecolor='none')

# for i in range(len(ratio_125)):
#     if i%2 == 0:
#         plt.plot(9 - 0.1, ratio_125[i], linestyle='none', marker='o', fillstyle='full',
#                  markeredgewidth=marker_edge_width, markersize=marker_size,
#                  markeredgecolor='black', markerfacecolor='none')
#     else:
#         plt.plot(9 + 0.1, ratio_125[i], linestyle='none', marker='o', fillstyle='full',
#                  markeredgewidth=marker_edge_width, markersize=marker_size,
#                  markeredgecolor='black', markerfacecolor='none')

for i in range(len(ratio_130)):
    if i % 2 == 0:
        plt.plot(5 - 0.1, ratio_130[i], linestyle='none', marker='o', fillstyle='full',
                 markeredgewidth=marker_edge_width, markersize=marker_size,
                 markeredgecolor='black', markerfacecolor='none')
    else:
        plt.plot(5 + 0.1, ratio_130[i], linestyle='none', marker='o', fillstyle='full',
                 markeredgewidth=marker_edge_width, markersize=marker_size,
                 markeredgecolor='black', markerfacecolor='none')

# for i in range(len(ratio_135)):
#     if i%2 == 0:
#         plt.plot(11 - 0.1, ratio_135[i], linestyle='none', marker='o', fillstyle='full',
#                  markeredgewidth=marker_edge_width, markersize=marker_size,
#                  markeredgecolor='black', markerfacecolor='none')
#     else:
#         plt.plot(11 + 0.1, ratio_135[i], linestyle='none', marker='o', fillstyle='full',
#                  markeredgewidth=marker_edge_width, markersize=marker_size,
#                  markeredgecolor='black', markerfacecolor='none')

for i in range(len(ratio_140)):
    if i % 2 == 0:
        plt.plot(6 - 0.1, ratio_140[i], linestyle='none', marker='o', fillstyle='full',
                 markeredgewidth=marker_edge_width, markersize=marker_size,
                 markeredgecolor='black', markerfacecolor='none')
    else:
        plt.plot(6 + 0.1, ratio_140[i], linestyle='none', marker='o', fillstyle='full',
                 markeredgewidth=marker_edge_width, markersize=marker_size,
                 markeredgecolor='black', markerfacecolor='none')

plt.xticks([0, 2, 4, 6], ['8/30', '10/30', '12/30', '14/30'], fontsize=font_size_1, **hfont)
# plt.xticks([0, 1, 2, 3], ['8/30', '8.5/30', '9/30', '10/30'], fontsize=font_size_1, **hfont)
plt.yticks([0, 1, 2, 3, 4, 5], fontsize=font_size_1, **hfont)
plt.xlabel('Feedforward input', fontsize=font_size_1, **hfont)
plt.ylabel('Fixed point to baseline ratio', fontsize=font_size_1, **hfont)
plt.xlim([-0.5, 6.5])
# plt.xlim([-0.5, 12.5])
plt.ylim([0, 5])
plt.hlines(y=1, xmin=-0.5, xmax=6.5, colors='k', linestyles=[(0, (6, 6, 6, 6))], linewidth=line_width)
plt.savefig('paper_figures/png/Revision_Fig_Point_1_2_Unstimulated_cotuned_neuron_SNN.png')
plt.savefig('paper_figures/pdf/Revision_Fig_Point_1_2_Unstimulated_cotuned_neuron_SNN.pdf')