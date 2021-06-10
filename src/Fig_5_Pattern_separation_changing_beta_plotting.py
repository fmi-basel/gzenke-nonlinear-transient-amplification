import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sympy.solvers import solve
from sympy import Symbol
from matplotlib import patches
import matplotlib.patches as mpatches
import scipy.io as sio
import math

# plotting configuration
ratio = 1.5
figure_len, figure_width = 15*ratio, 12*ratio
font_size_1, font_size_2 = 36*ratio, 36*ratio
legend_size = 18*ratio
line_width, tick_len = 3*ratio, 10*ratio
marker_size = 30*ratio
plot_line_width = 5*ratio
hfont = {'fontname': 'Arial'}
marker_edge_width = 4

pal = sns.color_palette("deep")

U_max = 6

l_beta = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

l_peak_E1_EE_STP, l_peak_E12_EE_STP, l_peak_E2_EE_STP, l_ss_E1_EE_STP, l_ss_E12_EE_STP, l_ss_E2_EE_STP = [], [], [], [], [], []
l_peak_E1_EI_STP, l_peak_E12_EI_STP, l_peak_E2_EI_STP, l_ss_E1_EI_STP, l_ss_E12_EI_STP, l_ss_E2_EI_STP = [], [], [], [], [], []

l_bs_E2_EE_STP, l_bs_E2_EI_STP = [], []

for beta in l_beta:
    l_r_e_1_2_EE_STP = sio.loadmat('data/Fig_5_Pattern_separation_activity_EE_STP_E12_beta_' + str(beta) + '.mat')['E12'][0]
    l_r_e_2_EE_STP = sio.loadmat('data/Fig_5_Pattern_separation_activity_EE_STP_E2_beta_' + str(beta) + '.mat')['E2'][0]

    l_r_e_1_2_EI_STP = sio.loadmat('data/Fig_5_Pattern_separation_activity_EI_STP_E12_beta_' + str(beta) + '_U_max_' + str(U_max) + '.mat')['E12'][0]
    l_r_e_2_EI_STP = sio.loadmat('data/Fig_5_Pattern_separation_activity_EI_STP_E2_beta_' + str(beta) + '_U_max_' + str(U_max) + '.mat')['E2'][0]

    l_peak_E1_EE_STP.append(np.nanmax(l_r_e_1_2_EE_STP[90000:110000]))
    l_ss_E1_EE_STP.append(np.nanmean(l_r_e_1_2_EE_STP[105000:109000]))
    l_peak_E12_EE_STP.append(np.nanmax(l_r_e_1_2_EE_STP[50000:70000]))
    l_ss_E12_EE_STP.append(np.nanmean(l_r_e_1_2_EE_STP[65000:69000]))
    l_peak_E2_EE_STP.append(np.nanmax(l_r_e_2_EE_STP[90000:110000]))
    l_ss_E2_EE_STP.append(np.nanmean(l_r_e_2_EE_STP[105000:109000]))

    l_peak_E1_EI_STP.append(np.nanmax(l_r_e_1_2_EI_STP[90000:110000]))
    l_ss_E1_EI_STP.append(np.nanmean(l_r_e_1_2_EI_STP[105000:109000]))
    l_peak_E12_EI_STP.append(np.nanmax(l_r_e_1_2_EI_STP[50000:70000]))
    l_ss_E12_EI_STP.append(np.nanmean(l_r_e_1_2_EI_STP[65000:69000]))
    l_peak_E2_EI_STP.append(np.nanmax(l_r_e_2_EI_STP[90000:110000]))
    l_ss_E2_EI_STP.append(np.nanmean(l_r_e_2_EI_STP[105000:109000]))

    l_bs_E2_EE_STP.append(np.nanmean(l_r_e_2_EE_STP[40000:49000]))
    l_bs_E2_EI_STP.append(np.nanmean(l_r_e_2_EI_STP[40000:49000]))


l_asso_peak_EE_STP, l_asso_peak_EI_STP, l_asso_ss_EE_STP, l_asso_ss_EI_STP = [], [], [], []
l_sepa_peak_EE_STP, l_sepa_peak_EI_STP, l_sepa_ss_EE_STP, l_sepa_ss_EI_STP = [], [], [], []
l_dis_peak_EE_STP, l_dis_peak_EI_STP, l_dis_ss_EE_STP, l_dis_ss_EI_STP = [], [], [], []

for i in range(len(l_peak_E1_EE_STP)):
    l_asso_peak_EE_STP.append(1 + (l_peak_E12_EE_STP[i] - l_peak_E1_EE_STP[i])/(l_peak_E1_EE_STP[i] + l_peak_E12_EE_STP[i]))
    l_asso_peak_EI_STP.append(1 + (l_peak_E12_EI_STP[i] - l_peak_E1_EI_STP[i])/(l_peak_E1_EI_STP[i] + l_peak_E12_EI_STP[i]))
    l_asso_ss_EE_STP.append(1 + (l_ss_E12_EE_STP[i] - l_ss_E1_EE_STP[i])/(l_ss_E1_EE_STP[i] + l_ss_E12_EE_STP[i]))
    l_asso_ss_EI_STP.append(1 + (l_ss_E12_EI_STP[i] - l_ss_E1_EI_STP[i])/(l_ss_E1_EI_STP[i] + l_ss_E12_EI_STP[i]))

    l_sepa_peak_EE_STP.append((l_peak_E1_EE_STP[i] - l_peak_E2_EE_STP[i])/(l_peak_E1_EE_STP[i] + l_peak_E2_EE_STP[i]))
    l_sepa_peak_EI_STP.append((l_peak_E1_EI_STP[i] - l_peak_E2_EI_STP[i])/(l_peak_E1_EI_STP[i] + l_peak_E2_EI_STP[i]))
    l_sepa_ss_EE_STP.append((l_ss_E1_EE_STP[i] - l_ss_E2_EE_STP[i])/(l_ss_E1_EE_STP[i] + l_ss_E2_EE_STP[i]))
    l_sepa_ss_EI_STP.append((l_ss_E1_EI_STP[i] - l_ss_E2_EI_STP[i])/(l_ss_E1_EI_STP[i] + l_ss_E2_EI_STP[i]))

    l_dis_peak_EE_STP.append(math.sin(math.radians(45 - round(math.degrees(
        math.asin(l_peak_E2_EE_STP[i] / np.sqrt(np.power(l_peak_E1_EE_STP[i], 2) + np.power(l_peak_E2_EE_STP[i], 2)))),
                                                               2))) * np.sqrt(
        np.power(l_peak_E1_EE_STP[i], 2) + np.power(l_peak_E2_EE_STP[i], 2)))
    l_dis_peak_EI_STP.append(math.sin(math.radians(45 - round(math.degrees(
        math.asin(l_peak_E2_EI_STP[i] / np.sqrt(np.power(l_peak_E1_EI_STP[i], 2) + np.power(l_peak_E2_EI_STP[i], 2)))),
                                                               2))) * np.sqrt(
        np.power(l_peak_E1_EI_STP[i], 2) + np.power(l_peak_E2_EI_STP[i], 2)))
    l_dis_ss_EE_STP.append(math.sin(math.radians(45 - round(math.degrees(
        math.asin(l_ss_E2_EE_STP[i] / np.sqrt(np.power(l_ss_E1_EE_STP[i], 2) + np.power(l_ss_E2_EE_STP[i], 2)))),
                                                             2))) * np.sqrt(
        np.power(l_ss_E1_EE_STP[i], 2) + np.power(l_ss_E2_EE_STP[i], 2)))
    l_dis_ss_EI_STP.append(math.sin(math.radians(45 - round(math.degrees(
        math.asin(l_ss_E2_EI_STP[i] / np.sqrt(np.power(l_ss_E1_EI_STP[i], 2) + np.power(l_ss_E2_EI_STP[i], 2)))),
                                                             2))) * np.sqrt(
        np.power(l_ss_E1_EI_STP[i], 2) + np.power(l_ss_E2_EI_STP[i], 2)))




plt.figure(figsize=(figure_len, figure_width))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(line_width)
plt.tick_params(width=line_width, length=tick_len)

plt.plot(l_asso_peak_EE_STP, color='gray', linewidth=plot_line_width)
plt.plot(l_asso_ss_EE_STP, color='gray', linestyle='dashed', linewidth=plot_line_width)

for i in range(len(l_peak_E1_EE_STP)):

    plt.plot(i, l_asso_peak_EE_STP[i], linestyle='none', marker='o', fillstyle='full',
                                             markeredgewidth=marker_edge_width, markersize=marker_size,
                                             markeredgecolor='black', markerfacecolor='gray')
    plt.plot(i, l_asso_ss_EE_STP[i], linestyle='none', marker='o', fillstyle='full',
                                             markeredgewidth=marker_edge_width, markersize=marker_size,
                                             markeredgecolor='black', markerfacecolor='gray')

plt.xticks([0, 2, 4, 6, 8, 10], [0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=font_size_1, **hfont)
plt.yticks([0, 0.5, 1.0], fontsize=font_size_1, **hfont)
plt.xlabel(r'$\beta$', fontsize=font_size_1, **hfont)
plt.ylabel('Association index', fontsize=font_size_1, **hfont)
plt.ylim([-0.05, 1.05])
plt.legend(['E-to-E STD onset transients', 'E-to-E STD fixed point'], prop={"family": "Arial", 'size': font_size_1}, loc='lower right')
plt.savefig('paper_figures/png/Fig_5_asso_index_peak_ss_changing_beta_U_max_' + str(U_max) + '_EE_STD.png')
plt.savefig('paper_figures/pdf/Fig_5_asso_index_peak_ss_changing_beta_U_max_' + str(U_max) + '_EE_STD.pdf')


plt.figure(figsize=(figure_len, figure_width))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(line_width)
plt.tick_params(width=line_width, length=tick_len)

plt.plot(l_asso_peak_EI_STP, color='m', linewidth=plot_line_width)
plt.plot(l_asso_ss_EI_STP, color='m', linestyle='dashed', linewidth=plot_line_width)

for i in range(len(l_peak_E1_EE_STP)):

    plt.plot(i, l_asso_peak_EI_STP[i], linestyle='none', marker='o', fillstyle='full',
                                             markeredgewidth=marker_edge_width, markersize=marker_size,
                                             markeredgecolor='black', markerfacecolor='m')
    plt.plot(i, l_asso_ss_EI_STP[i], linestyle='none', marker='o', fillstyle='full',
                                             markeredgewidth=marker_edge_width, markersize=marker_size,
                                             markeredgecolor='black', markerfacecolor='m')

plt.xticks([0, 2, 4, 6, 8, 10], [0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=font_size_1, **hfont)
plt.yticks([0, 0.5, 1.0], fontsize=font_size_1, **hfont)
plt.xlabel(r'$\beta$', fontsize=font_size_1, **hfont)
plt.ylabel('Association index', fontsize=font_size_1, **hfont)
plt.ylim([-0.05, 1.05])
plt.legend(['E-to-I STF onset transients', 'E-to-I STF fixed point'], prop={"family": "Arial", 'size': font_size_1}, loc='lower right')
plt.savefig('paper_figures/png/Fig_5_asso_index_peak_ss_changing_beta_U_max_' + str(U_max) + '_EI_STF.png')
plt.savefig('paper_figures/pdf/Fig_5_asso_index_peak_ss_changing_beta_U_max_' + str(U_max) + '_EI_STF.pdf')


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

plt.plot(l_dis_peak_EE_STP, color='gray', linewidth=plot_line_width)
plt.plot(l_dis_ss_EE_STP, color='gray', linestyle='dashed', linewidth=plot_line_width)


for i in range(len(l_peak_E1_EE_STP)):

    plt.plot(i, l_dis_peak_EE_STP[i], linestyle='none', marker='o', fillstyle='full',
                                             markeredgewidth=marker_edge_width, markersize=marker_size,
                                             markeredgecolor='black', markerfacecolor='gray')#, alpha=0.3+0.06*i)

    plt.plot(i, l_dis_ss_EE_STP[i], linestyle='none', marker='o', fillstyle='full',
                                             markeredgewidth=marker_edge_width, markersize=marker_size,
                                             markeredgecolor='black', markerfacecolor='gray')#, alpha=0.3+0.06*i)


plt.xticks([0, 2, 4, 6, 8, 10], [0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=font_size_1, **hfont)
plt.yticks([0, 1, 100, 10000, 1000000], fontsize=font_size_1, **hfont)
plt.xlabel(r'$\beta$', fontsize=font_size_1, **hfont)
plt.ylabel('Distance to the decision boundary', fontsize=font_size_1, **hfont)
plt.ylim([0, 1000000])
plt.legend(['E-to-E STD onset transients', 'E-to-E STD fixed point'], prop={"family": "Arial", 'size': font_size_1}, loc='lower right')
plt.savefig('paper_figures/png/Fig_5_sepa_dis_EE_STP_changing_beta_U_max_' + str(U_max) + '.png')
plt.savefig('paper_figures/pdf/Fig_5_sepa_dis_EE_STP_changing_beta_U_max_' + str(U_max) + '.pdf')


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

plt.plot(l_dis_peak_EI_STP, color='m', linewidth=plot_line_width)
plt.plot(l_dis_ss_EI_STP, color='m', linestyle='dashed', linewidth=plot_line_width)

for i in range(len(l_peak_E1_EE_STP)):

    plt.plot(i, l_dis_peak_EI_STP[i], linestyle='none', marker='o', fillstyle='full',
                                             markeredgewidth=marker_edge_width, markersize=marker_size,
                                             markeredgecolor='black', markerfacecolor='m')#, alpha=0.3+0.06*i)
    plt.plot(i, l_dis_ss_EI_STP[i], linestyle='none', marker='o', fillstyle='full',
                                             markeredgewidth=marker_edge_width, markersize=marker_size,
                                             markeredgecolor='black', markerfacecolor='m')#, alpha=0.3+0.06*i)

plt.xticks([0, 2, 4, 6, 8, 10], [0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=font_size_1, **hfont)
plt.yticks([0, 1, 100, 10000, 1000000], fontsize=font_size_1, **hfont)
plt.xlabel(r'$\beta$', fontsize=font_size_1, **hfont)
plt.ylabel('Distance to the decision boundary', fontsize=font_size_1, **hfont)
plt.ylim([0, 1000000])
plt.legend(['E-to-I STF onset transients', 'E-to-I STF fixed point'], prop={"family": "Arial", 'size': font_size_1}, loc='lower right')
plt.savefig('paper_figures/png/Fig_5_sepa_dis_EI_STP_changing_beta_U_max_' + str(U_max) + '.png')
plt.savefig('paper_figures/pdf/Fig_5_sepa_dis_EI_STP_changing_beta_U_max_' + str(U_max) + '.pdf')


