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
l_color = ['#D6EAF8', '#85C1E9', '#3498DB', '#2874A6'] # ['#D6EAF8', '#85C1E9', '#3498DB', '#2874A6']

U_max = 6

l_p = [0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0]

l_peak_E1_EE_STP, l_peak_E2_EE_STP, l_ss_E1_EE_STP, l_ss_E2_EE_STP = [], [], [], []
l_peak_E1_EI_STP, l_peak_E2_EI_STP, l_ss_E1_EI_STP, l_ss_E2_EI_STP = [], [], [], []

l_peak_E1_EE_STP_2, l_peak_E2_EE_STP_2, l_ss_E1_EE_STP_2, l_ss_E2_EE_STP_2 = [], [], [], []
l_peak_E1_EI_STP_2, l_peak_E2_EI_STP_2, l_ss_E1_EI_STP_2, l_ss_E2_EI_STP_2 = [], [], [], []

l_peak_E1_EE_STP_3, l_peak_E2_EE_STP_3, l_ss_E1_EE_STP_3, l_ss_E2_EE_STP_3 = [], [], [], []
l_peak_E1_EI_STP_3, l_peak_E2_EI_STP_3, l_ss_E1_EI_STP_3, l_ss_E2_EI_STP_3 = [], [], [], []

l_peak_E1_EE_STP_0, l_peak_E2_EE_STP_0, l_ss_E1_EE_STP_0, l_ss_E2_EE_STP_0 = [], [], [], []
l_peak_E1_EI_STP_0, l_peak_E2_EI_STP_0, l_ss_E1_EI_STP_0, l_ss_E2_EI_STP_0 = [], [], [], []

l_bs_E2_EE_STP, l_bs_E2_EI_STP = [], []

s_path = '../Redo_part/'

for p in l_p:
    l_r_e_1_EE_STP = sio.loadmat(s_path + 'data/Fig_6_Morphing_activity_EE_STP_E1_Jee_1.1_p_' + str(p) + '.mat')['E1'][0]
    l_r_e_2_EE_STP = sio.loadmat(s_path + 'data/Fig_6_Morphing_activity_EE_STP_E2_Jee_1.1_p_' + str(p) + '.mat')['E2'][0]
    l_r_e_1_EI_STP = sio.loadmat(s_path + 'data/Fig_6_Morphing_activity_EI_STP_E1_Jee_1.1_p_' + str(p) + '.mat')['E1'][0]
    l_r_e_2_EI_STP = sio.loadmat(s_path + 'data/Fig_6_Morphing_activity_EI_STP_E2_Jee_1.1_p_' + str(p) + '.mat')['E2'][0]

    l_r_e_1_EE_STP_2 = sio.loadmat(s_path + 'data/Fig_6_Morphing_activity_EE_STP_E1_Jee_1.2_p_' + str(p) + '.mat')['E1'][0]
    l_r_e_2_EE_STP_2 = sio.loadmat(s_path + 'data/Fig_6_Morphing_activity_EE_STP_E2_Jee_1.2_p_' + str(p) + '.mat')['E2'][0]
    l_r_e_1_EI_STP_2 = sio.loadmat(s_path + 'data/Fig_6_Morphing_activity_EI_STP_E1_Jee_1.2_p_' + str(p) + '.mat')['E1'][0]
    l_r_e_2_EI_STP_2 = sio.loadmat(s_path + 'data/Fig_6_Morphing_activity_EI_STP_E2_Jee_1.2_p_' + str(p) + '.mat')['E2'][0]

    l_r_e_1_EE_STP_3 = sio.loadmat(s_path + 'data/Fig_6_Morphing_activity_EE_STP_E1_Jee_1.3_p_' + str(p) + '.mat')['E1'][0]
    l_r_e_2_EE_STP_3 = sio.loadmat(s_path + 'data/Fig_6_Morphing_activity_EE_STP_E2_Jee_1.3_p_' + str(p) + '.mat')['E2'][0]
    l_r_e_1_EI_STP_3 = sio.loadmat(s_path + 'data/Fig_6_Morphing_activity_EI_STP_E1_Jee_1.3_p_' + str(p) + '.mat')['E1'][0]
    l_r_e_2_EI_STP_3 = sio.loadmat(s_path + 'data/Fig_6_Morphing_activity_EI_STP_E2_Jee_1.3_p_' + str(p) + '.mat')['E2'][0]

    l_r_e_1_EE_STP_0 = sio.loadmat(s_path + 'data/Fig_6_Morphing_activity_EE_STP_E1_Jee_0.0_p_' + str(p) + '.mat')['E1'][0]
    l_r_e_2_EE_STP_0 = sio.loadmat(s_path + 'data/Fig_6_Morphing_activity_EE_STP_E2_Jee_0.0_p_' + str(p) + '.mat')['E2'][0]
    l_r_e_1_EI_STP_0 = sio.loadmat(s_path + 'data/Fig_6_Morphing_activity_EI_STP_E1_Jee_0.0_p_' + str(p) + '.mat')['E1'][0]
    l_r_e_2_EI_STP_0 = sio.loadmat(s_path + 'data/Fig_6_Morphing_activity_EI_STP_E2_Jee_0.0_p_' + str(p) + '.mat')['E2'][0]


    l_peak_E1_EE_STP.append(np.nanmax(l_r_e_1_EE_STP[50000:70000]))
    l_ss_E1_EE_STP.append(np.nanmean(l_r_e_1_EE_STP[65000:69000]))
    l_peak_E2_EE_STP.append(np.nanmax(l_r_e_2_EE_STP[50000:70000]))
    l_ss_E2_EE_STP.append(np.nanmean(l_r_e_2_EE_STP[65000:69000]))

    l_peak_E1_EI_STP.append(np.nanmax(l_r_e_1_EI_STP[50000:70000]))
    l_ss_E1_EI_STP.append(np.nanmean(l_r_e_1_EI_STP[65000:69000]))
    l_peak_E2_EI_STP.append(np.nanmax(l_r_e_2_EI_STP[50000:70000]))
    l_ss_E2_EI_STP.append(np.nanmean(l_r_e_2_EI_STP[65000:69000]))


    l_peak_E1_EE_STP_2.append(np.nanmax(l_r_e_1_EE_STP_2[50000:70000]))
    l_ss_E1_EE_STP_2.append(np.nanmean(l_r_e_1_EE_STP_2[65000:69000]))
    l_peak_E2_EE_STP_2.append(np.nanmax(l_r_e_2_EE_STP_2[50000:70000]))
    l_ss_E2_EE_STP_2.append(np.nanmean(l_r_e_2_EE_STP_2[65000:69000]))

    l_peak_E1_EI_STP_2.append(np.nanmax(l_r_e_1_EI_STP_2[50000:70000]))
    l_ss_E1_EI_STP_2.append(np.nanmean(l_r_e_1_EI_STP_2[65000:69000]))
    l_peak_E2_EI_STP_2.append(np.nanmax(l_r_e_2_EI_STP_2[50000:70000]))
    l_ss_E2_EI_STP_2.append(np.nanmean(l_r_e_2_EI_STP_2[65000:69000]))


    l_peak_E1_EE_STP_3.append(np.nanmax(l_r_e_1_EE_STP_3[50000:70000]))
    l_ss_E1_EE_STP_3.append(np.nanmean(l_r_e_1_EE_STP_3[65000:69000]))
    l_peak_E2_EE_STP_3.append(np.nanmax(l_r_e_2_EE_STP_3[50000:70000]))
    l_ss_E2_EE_STP_3.append(np.nanmean(l_r_e_2_EE_STP_3[65000:69000]))

    l_peak_E1_EI_STP_3.append(np.nanmax(l_r_e_1_EI_STP_3[50000:70000]))
    l_ss_E1_EI_STP_3.append(np.nanmean(l_r_e_1_EI_STP_3[65000:69000]))
    l_peak_E2_EI_STP_3.append(np.nanmax(l_r_e_2_EI_STP_3[50000:70000]))
    l_ss_E2_EI_STP_3.append(np.nanmean(l_r_e_2_EI_STP_3[65000:69000]))


    l_peak_E1_EE_STP_0.append(np.nanmax(l_r_e_1_EE_STP_0[50000:70000]))
    l_ss_E1_EE_STP_0.append(np.nanmean(l_r_e_1_EE_STP_0[65000:69000]))
    l_peak_E2_EE_STP_0.append(np.nanmax(l_r_e_2_EE_STP_0[50000:70000]))
    l_ss_E2_EE_STP_0.append(np.nanmean(l_r_e_2_EE_STP_0[65000:69000]))

    l_peak_E1_EI_STP_0.append(np.nanmax(l_r_e_1_EI_STP_0[50000:70000]))
    l_ss_E1_EI_STP_0.append(np.nanmean(l_r_e_1_EI_STP_0[65000:69000]))
    l_peak_E2_EI_STP_0.append(np.nanmax(l_r_e_2_EI_STP_0[50000:70000]))
    l_ss_E2_EI_STP_0.append(np.nanmean(l_r_e_2_EI_STP_0[65000:69000]))


l_idx_peak_EE_STP, l_idx_peak_EI_STP, l_idx_ss_EE_STP, l_idx_ss_EI_STP = [], [], [], []
l_idx_peak_EE_STP_2, l_idx_peak_EI_STP_2, l_idx_ss_EE_STP_2, l_idx_ss_EI_STP_2 = [], [], [], []
l_idx_peak_EE_STP_3, l_idx_peak_EI_STP_3, l_idx_ss_EE_STP_3, l_idx_ss_EI_STP_3 = [], [], [], []
l_idx_peak_EE_STP_0, l_idx_peak_EI_STP_0, l_idx_ss_EE_STP_0, l_idx_ss_EI_STP_0 = [], [], [], []


for i in range(len(l_peak_E1_EE_STP)):

    l_idx_peak_EE_STP.append((l_peak_E1_EE_STP[i]-l_peak_E2_EE_STP[i])/(l_peak_E1_EE_STP[i]+l_peak_E2_EE_STP[i]))
    l_idx_peak_EI_STP.append((l_peak_E1_EI_STP[i]-l_peak_E2_EI_STP[i])/(l_peak_E1_EI_STP[i]+l_peak_E2_EI_STP[i]))
    l_idx_ss_EE_STP.append((l_ss_E1_EE_STP[i]-l_ss_E2_EE_STP[i])/(l_ss_E1_EE_STP[i]+l_ss_E2_EE_STP[i]))
    l_idx_ss_EI_STP.append((l_ss_E1_EI_STP[i]-l_ss_E2_EI_STP[i])/(l_ss_E1_EI_STP[i]+l_ss_E2_EI_STP[i]))


    l_idx_peak_EE_STP_2.append((l_peak_E1_EE_STP_2[i]-l_peak_E2_EE_STP_2[i])/(l_peak_E1_EE_STP_2[i]+l_peak_E2_EE_STP_2[i]))
    l_idx_peak_EI_STP_2.append((l_peak_E1_EI_STP_2[i]-l_peak_E2_EI_STP_2[i])/(l_peak_E1_EI_STP_2[i]+l_peak_E2_EI_STP_2[i]))
    l_idx_ss_EE_STP_2.append((l_ss_E1_EE_STP_2[i]-l_ss_E2_EE_STP_2[i])/(l_ss_E1_EE_STP_2[i]+l_ss_E2_EE_STP_2[i]))
    l_idx_ss_EI_STP_2.append((l_ss_E1_EI_STP_2[i]-l_ss_E2_EI_STP_2[i])/(l_ss_E1_EI_STP_2[i]+l_ss_E2_EI_STP_2[i]))

    l_idx_peak_EE_STP_3.append((l_peak_E1_EE_STP_3[i]-l_peak_E2_EE_STP_3[i])/(l_peak_E1_EE_STP_3[i]+l_peak_E2_EE_STP_3[i]))
    l_idx_peak_EI_STP_3.append((l_peak_E1_EI_STP_3[i]-l_peak_E2_EI_STP_3[i])/(l_peak_E1_EI_STP_3[i]+l_peak_E2_EI_STP_3[i]))
    l_idx_ss_EE_STP_3.append((l_ss_E1_EE_STP_3[i]-l_ss_E2_EE_STP_3[i])/(l_ss_E1_EE_STP_3[i]+l_ss_E2_EE_STP_3[i]))
    l_idx_ss_EI_STP_3.append((l_ss_E1_EI_STP_3[i]-l_ss_E2_EI_STP_3[i])/(l_ss_E1_EI_STP_3[i]+l_ss_E2_EI_STP_3[i]))

    l_idx_peak_EE_STP_0.append((l_peak_E1_EE_STP_0[i]-l_peak_E2_EE_STP_0[i])/(l_peak_E1_EE_STP_0[i]+l_peak_E2_EE_STP_0[i]))
    l_idx_peak_EI_STP_0.append((l_peak_E1_EI_STP_0[i]-l_peak_E2_EI_STP_0[i])/(l_peak_E1_EI_STP_0[i]+l_peak_E2_EI_STP_0[i]))
    l_idx_ss_EE_STP_0.append((l_ss_E1_EE_STP_0[i]-l_ss_E2_EE_STP_0[i])/(l_ss_E1_EE_STP_0[i]+l_ss_E2_EE_STP_0[i]))
    l_idx_ss_EI_STP_0.append((l_ss_E1_EI_STP_0[i]-l_ss_E2_EI_STP_0[i])/(l_ss_E1_EI_STP_0[i]+l_ss_E2_EI_STP_0[i]))


# l_dis_peak_EE_STP, l_dis_peak_EI_STP, l_dis_ss_EE_STP, l_dis_ss_EI_STP = [], [], [], []
# l_dis_peak_EE_STP_2, l_dis_peak_EI_STP_2, l_dis_ss_EE_STP_2, l_dis_ss_EI_STP_2 = [], [], [], []
# l_dis_peak_EE_STP_3, l_dis_peak_EI_STP_3, l_dis_ss_EE_STP_3, l_dis_ss_EI_STP_3 = [], [], [], []
# l_dis_peak_EE_STP_0, l_dis_peak_EI_STP_0, l_dis_ss_EE_STP_0, l_dis_ss_EI_STP_0 = [], [], [], []
#
# for i in range(len(l_peak_E1_EE_STP)):
#
#     if i < len(l_p)/2:
#         l_dis_peak_EE_STP.append(math.sin(math.radians(45 - round(math.degrees(
#             math.asin(l_peak_E1_EE_STP[i] / np.sqrt(np.power(l_peak_E1_EE_STP[i], 2) + np.power(l_peak_E2_EE_STP[i], 2)))),
#                                                                    2))) * np.sqrt(
#             np.power(l_peak_E1_EE_STP[i], 2) + np.power(l_peak_E2_EE_STP[i], 2)))
#         l_dis_peak_EI_STP.append(math.sin(math.radians(45 - round(math.degrees(
#             math.asin(l_peak_E1_EI_STP[i] / np.sqrt(np.power(l_peak_E1_EI_STP[i], 2) + np.power(l_peak_E2_EI_STP[i], 2)))),
#                                                                    2))) * np.sqrt(
#             np.power(l_peak_E1_EI_STP[i], 2) + np.power(l_peak_E2_EI_STP[i], 2)))
#         l_dis_ss_EE_STP.append(math.sin(math.radians(45 - round(math.degrees(
#             math.asin(l_ss_E1_EE_STP[i] / np.sqrt(np.power(l_ss_E1_EE_STP[i], 2) + np.power(l_ss_E2_EE_STP[i], 2)))),
#                                                                  2))) * np.sqrt(
#             np.power(l_ss_E1_EE_STP[i], 2) + np.power(l_ss_E2_EE_STP[i], 2)))
#         l_dis_ss_EI_STP.append(math.sin(math.radians(45 - round(math.degrees(
#             math.asin(l_ss_E1_EI_STP[i] / np.sqrt(np.power(l_ss_E1_EI_STP[i], 2) + np.power(l_ss_E2_EI_STP[i], 2)))),
#                                                                  2))) * np.sqrt(
#             np.power(l_ss_E1_EI_STP[i], 2) + np.power(l_ss_E2_EI_STP[i], 2)))
#
#
#
#         l_dis_peak_EE_STP_2.append(math.sin(math.radians(45 - round(math.degrees(
#             math.asin(l_peak_E1_EE_STP_2[i] / np.sqrt(np.power(l_peak_E1_EE_STP_2[i], 2) + np.power(l_peak_E2_EE_STP_2[i], 2)))),
#                                                                    2))) * np.sqrt(
#             np.power(l_peak_E1_EE_STP_2[i], 2) + np.power(l_peak_E2_EE_STP_2[i], 2)))
#         l_dis_peak_EI_STP_2.append(math.sin(math.radians(45 - round(math.degrees(
#             math.asin(l_peak_E1_EI_STP_2[i] / np.sqrt(np.power(l_peak_E1_EI_STP_2[i], 2) + np.power(l_peak_E2_EI_STP_2[i], 2)))),
#                                                                    2))) * np.sqrt(
#             np.power(l_peak_E1_EI_STP_2[i], 2) + np.power(l_peak_E2_EI_STP_2[i], 2)))
#         l_dis_ss_EE_STP_2.append(math.sin(math.radians(45 - round(math.degrees(
#             math.asin(l_ss_E1_EE_STP_2[i] / np.sqrt(np.power(l_ss_E1_EE_STP_2[i], 2) + np.power(l_ss_E2_EE_STP_2[i], 2)))),
#                                                                  2))) * np.sqrt(
#             np.power(l_ss_E1_EE_STP_2[i], 2) + np.power(l_ss_E2_EE_STP_2[i], 2)))
#         l_dis_ss_EI_STP_2.append(math.sin(math.radians(45 - round(math.degrees(
#             math.asin(l_ss_E1_EI_STP_2[i] / np.sqrt(np.power(l_ss_E1_EI_STP_2[i], 2) + np.power(l_ss_E2_EI_STP_2[i], 2)))),
#                                                                  2))) * np.sqrt(
#             np.power(l_ss_E1_EI_STP_2[i], 2) + np.power(l_ss_E2_EI_STP_2[i], 2)))
#
#
#
#         l_dis_peak_EE_STP_3.append(math.sin(math.radians(45 - round(math.degrees(
#             math.asin(l_peak_E1_EE_STP_3[i] / np.sqrt(np.power(l_peak_E1_EE_STP_3[i], 2) + np.power(l_peak_E2_EE_STP_3[i], 2)))),
#                                                                    2))) * np.sqrt(
#             np.power(l_peak_E1_EE_STP_3[i], 2) + np.power(l_peak_E2_EE_STP_3[i], 2)))
#         l_dis_peak_EI_STP_3.append(math.sin(math.radians(45 - round(math.degrees(
#             math.asin(l_peak_E1_EI_STP_3[i] / np.sqrt(np.power(l_peak_E1_EI_STP_3[i], 2) + np.power(l_peak_E2_EI_STP_3[i], 2)))),
#                                                                    2))) * np.sqrt(
#             np.power(l_peak_E1_EI_STP_3[i], 2) + np.power(l_peak_E2_EI_STP_3[i], 2)))
#         l_dis_ss_EE_STP_3.append(math.sin(math.radians(45 - round(math.degrees(
#             math.asin(l_ss_E1_EE_STP_3[i] / np.sqrt(np.power(l_ss_E1_EE_STP_3[i], 2) + np.power(l_ss_E2_EE_STP_3[i], 2)))),
#                                                                  2))) * np.sqrt(
#             np.power(l_ss_E1_EE_STP_3[i], 2) + np.power(l_ss_E2_EE_STP_3[i], 2)))
#         l_dis_ss_EI_STP_3.append(math.sin(math.radians(45 - round(math.degrees(
#             math.asin(l_ss_E1_EI_STP_3[i] / np.sqrt(np.power(l_ss_E1_EI_STP_3[i], 2) + np.power(l_ss_E2_EI_STP_3[i], 2)))),
#                                                                  2))) * np.sqrt(
#             np.power(l_ss_E1_EI_STP_3[i], 2) + np.power(l_ss_E2_EI_STP_3[i], 2)))
#
#
#
#         l_dis_peak_EE_STP_0.append(math.sin(math.radians(45 - round(math.degrees(
#             math.asin(l_peak_E1_EE_STP_0[i] / np.sqrt(np.power(l_peak_E1_EE_STP_0[i], 2) + np.power(l_peak_E2_EE_STP_0[i], 2)))),
#                                                                    2))) * np.sqrt(
#             np.power(l_peak_E1_EE_STP_0[i], 2) + np.power(l_peak_E2_EE_STP_0[i], 2)))
#         l_dis_peak_EI_STP_0.append(math.sin(math.radians(45 - round(math.degrees(
#             math.asin(l_peak_E1_EI_STP_0[i] / np.sqrt(np.power(l_peak_E1_EI_STP_0[i], 2) + np.power(l_peak_E2_EI_STP_0[i], 2)))),
#                                                                    2))) * np.sqrt(
#             np.power(l_peak_E1_EI_STP_0[i], 2) + np.power(l_peak_E2_EI_STP_0[i], 2)))
#         l_dis_ss_EE_STP_0.append(math.sin(math.radians(45 - round(math.degrees(
#             math.asin(l_ss_E1_EE_STP_0[i] / np.sqrt(np.power(l_ss_E1_EE_STP_0[i], 2) + np.power(l_ss_E2_EE_STP_0[i], 2)))),
#                                                                  2))) * np.sqrt(
#             np.power(l_ss_E1_EE_STP_0[i], 2) + np.power(l_ss_E2_EE_STP_0[i], 2)))
#         l_dis_ss_EI_STP_0.append(math.sin(math.radians(45 - round(math.degrees(
#             math.asin(l_ss_E1_EI_STP_0[i] / np.sqrt(np.power(l_ss_E1_EI_STP_0[i], 2) + np.power(l_ss_E2_EI_STP_0[i], 2)))),
#                                                                  2))) * np.sqrt(
#             np.power(l_ss_E1_EI_STP_0[i], 2) + np.power(l_ss_E2_EI_STP_0[i], 2)))
#
#     else:
#         l_dis_peak_EE_STP.append(math.sin(math.radians(45 - round(math.degrees(
#             math.asin(l_peak_E2_EE_STP[i] / np.sqrt(np.power(l_peak_E1_EE_STP[i], 2) + np.power(l_peak_E2_EE_STP[i], 2)))),
#                                                                    2))) * np.sqrt(
#             np.power(l_peak_E1_EE_STP[i], 2) + np.power(l_peak_E2_EE_STP[i], 2)))
#         l_dis_peak_EI_STP.append(math.sin(math.radians(45 - round(math.degrees(
#             math.asin(l_peak_E2_EI_STP[i] / np.sqrt(np.power(l_peak_E1_EI_STP[i], 2) + np.power(l_peak_E2_EI_STP[i], 2)))),
#                                                                    2))) * np.sqrt(
#             np.power(l_peak_E1_EI_STP[i], 2) + np.power(l_peak_E2_EI_STP[i], 2)))
#         l_dis_ss_EE_STP.append(math.sin(math.radians(45 - round(math.degrees(
#             math.asin(l_ss_E2_EE_STP[i] / np.sqrt(np.power(l_ss_E1_EE_STP[i], 2) + np.power(l_ss_E2_EE_STP[i], 2)))),
#                                                                  2))) * np.sqrt(
#             np.power(l_ss_E1_EE_STP[i], 2) + np.power(l_ss_E2_EE_STP[i], 2)))
#         l_dis_ss_EI_STP.append(math.sin(math.radians(45 - round(math.degrees(
#             math.asin(l_ss_E2_EI_STP[i] / np.sqrt(np.power(l_ss_E1_EI_STP[i], 2) + np.power(l_ss_E2_EI_STP[i], 2)))),
#                                                                  2))) * np.sqrt(
#             np.power(l_ss_E1_EI_STP[i], 2) + np.power(l_ss_E2_EI_STP[i], 2)))
#
#
#
#         l_dis_peak_EE_STP_2.append(math.sin(math.radians(45 - round(math.degrees(
#             math.asin(l_peak_E2_EE_STP_2[i] / np.sqrt(np.power(l_peak_E1_EE_STP_2[i], 2) + np.power(l_peak_E2_EE_STP_2[i], 2)))),
#                                                                    2))) * np.sqrt(
#             np.power(l_peak_E1_EE_STP_2[i], 2) + np.power(l_peak_E2_EE_STP_2[i], 2)))
#         l_dis_peak_EI_STP_2.append(math.sin(math.radians(45 - round(math.degrees(
#             math.asin(l_peak_E2_EI_STP_2[i] / np.sqrt(np.power(l_peak_E1_EI_STP_2[i], 2) + np.power(l_peak_E2_EI_STP_2[i], 2)))),
#                                                                    2))) * np.sqrt(
#             np.power(l_peak_E1_EI_STP_2[i], 2) + np.power(l_peak_E2_EI_STP_2[i], 2)))
#         l_dis_ss_EE_STP_2.append(math.sin(math.radians(45 - round(math.degrees(
#             math.asin(l_ss_E2_EE_STP_2[i] / np.sqrt(np.power(l_ss_E1_EE_STP_2[i], 2) + np.power(l_ss_E2_EE_STP_2[i], 2)))),
#                                                                  2))) * np.sqrt(
#             np.power(l_ss_E1_EE_STP_2[i], 2) + np.power(l_ss_E2_EE_STP_2[i], 2)))
#         l_dis_ss_EI_STP_2.append(math.sin(math.radians(45 - round(math.degrees(
#             math.asin(l_ss_E2_EI_STP_2[i] / np.sqrt(np.power(l_ss_E1_EI_STP_2[i], 2) + np.power(l_ss_E2_EI_STP_2[i], 2)))),
#                                                                  2))) * np.sqrt(
#             np.power(l_ss_E1_EI_STP_2[i], 2) + np.power(l_ss_E2_EI_STP_2[i], 2)))
#
#
#
#         l_dis_peak_EE_STP_3.append(math.sin(math.radians(45 - round(math.degrees(
#             math.asin(l_peak_E2_EE_STP_3[i] / np.sqrt(np.power(l_peak_E1_EE_STP_3[i], 2) + np.power(l_peak_E2_EE_STP_3[i], 2)))),
#                                                                    2))) * np.sqrt(
#             np.power(l_peak_E1_EE_STP_3[i], 2) + np.power(l_peak_E2_EE_STP_3[i], 2)))
#         l_dis_peak_EI_STP_3.append(math.sin(math.radians(45 - round(math.degrees(
#             math.asin(l_peak_E2_EI_STP_3[i] / np.sqrt(np.power(l_peak_E1_EI_STP_3[i], 2) + np.power(l_peak_E2_EI_STP_3[i], 2)))),
#                                                                    2))) * np.sqrt(
#             np.power(l_peak_E1_EI_STP_3[i], 2) + np.power(l_peak_E2_EI_STP_3[i], 2)))
#         l_dis_ss_EE_STP_3.append(math.sin(math.radians(45 - round(math.degrees(
#             math.asin(l_ss_E2_EE_STP_3[i] / np.sqrt(np.power(l_ss_E1_EE_STP_3[i], 2) + np.power(l_ss_E2_EE_STP_3[i], 2)))),
#                                                                  2))) * np.sqrt(
#             np.power(l_ss_E1_EE_STP_3[i], 2) + np.power(l_ss_E2_EE_STP_3[i], 2)))
#         l_dis_ss_EI_STP_3.append(math.sin(math.radians(45 - round(math.degrees(
#             math.asin(l_ss_E2_EI_STP_3[i] / np.sqrt(np.power(l_ss_E1_EI_STP_3[i], 2) + np.power(l_ss_E2_EI_STP_3[i], 2)))),
#                                                                  2))) * np.sqrt(
#             np.power(l_ss_E1_EI_STP_3[i], 2) + np.power(l_ss_E2_EI_STP_3[i], 2)))
#
#
#         l_dis_peak_EE_STP_0.append(math.sin(math.radians(45 - round(math.degrees(
#             math.asin(l_peak_E2_EE_STP_0[i] / np.sqrt(np.power(l_peak_E1_EE_STP_0[i], 2) + np.power(l_peak_E2_EE_STP_0[i], 2)))),
#                                                                    2))) * np.sqrt(
#             np.power(l_peak_E1_EE_STP_0[i], 2) + np.power(l_peak_E2_EE_STP_0[i], 2)))
#         l_dis_peak_EI_STP_0.append(math.sin(math.radians(45 - round(math.degrees(
#             math.asin(l_peak_E2_EI_STP_0[i] / np.sqrt(np.power(l_peak_E1_EI_STP_0[i], 2) + np.power(l_peak_E2_EI_STP_0[i], 2)))),
#                                                                    2))) * np.sqrt(
#             np.power(l_peak_E1_EI_STP_0[i], 2) + np.power(l_peak_E2_EI_STP_0[i], 2)))
#         l_dis_ss_EE_STP_0.append(math.sin(math.radians(45 - round(math.degrees(
#             math.asin(l_ss_E2_EE_STP_0[i] / np.sqrt(np.power(l_ss_E1_EE_STP_0[i], 2) + np.power(l_ss_E2_EE_STP_0[i], 2)))),
#                                                                  2))) * np.sqrt(
#             np.power(l_ss_E1_EE_STP_0[i], 2) + np.power(l_ss_E2_EE_STP_0[i], 2)))
#         l_dis_ss_EI_STP_0.append(math.sin(math.radians(45 - round(math.degrees(
#             math.asin(l_ss_E2_EI_STP_0[i] / np.sqrt(np.power(l_ss_E1_EI_STP_0[i], 2) + np.power(l_ss_E2_EI_STP_0[i], 2)))),
#                                                                  2))) * np.sqrt(
#             np.power(l_ss_E1_EI_STP_0[i], 2) + np.power(l_ss_E2_EI_STP_0[i], 2)))
#


plt.figure(figsize=(figure_len, figure_width))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(line_width)
plt.tick_params(width=line_width, length=tick_len)

plt.plot(l_idx_peak_EE_STP_0, color='blue', linewidth=plot_line_width, alpha=0.2)
plt.plot(l_idx_peak_EE_STP, color='blue', linewidth=plot_line_width, alpha=0.4)
plt.plot(l_idx_peak_EE_STP_2, color='blue', linewidth=plot_line_width, alpha=0.6)
plt.plot(l_idx_peak_EE_STP_3, color='blue', linewidth=plot_line_width, alpha=1.0)
plt.plot(l_idx_ss_EE_STP_0, color='blue', linestyle='dashed', linewidth=plot_line_width, alpha=0.2)
plt.plot(l_idx_ss_EE_STP, color='blue', linestyle='dashed', linewidth=plot_line_width, alpha=0.4)
plt.plot(l_idx_ss_EE_STP_2, color='blue', linestyle='dashed', linewidth=plot_line_width, alpha=0.6)
plt.plot(l_idx_ss_EE_STP_3, color='blue', linestyle='dashed', linewidth=plot_line_width, alpha=1.0)


plt.xticks([0, 10, 20, 30, 40], [0, 0.25, 0.5, 0.75, 1.0], fontsize=font_size_1, **hfont)
plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0], fontsize=font_size_1, **hfont)
plt.xlabel('$p$', fontsize=font_size_1, **hfont)
plt.ylabel('Separation index', fontsize=font_size_1, **hfont)
plt.ylim([-1.05, 1.05])
plt.legend([r"$J_{EE}$: 0.0", r"$J_{EE}$: 1.1", r"$J_{EE}$: 1.2", r"$J_{EE}$: 1.3"], prop={"family": "Arial", 'size': font_size_1}, loc='lower right')
plt.savefig('paper_figures/png/Revision_Fig_Point_2_9_Morphing_EE_STP_changing_Jee_U_max_' + str(U_max) + '.png')
plt.savefig('paper_figures/pdf/Revision_Fig_Point_2_9_Morphing_EE_STP_changing_Jee_U_max_' + str(U_max) + '.pdf')


plt.figure(figsize=(figure_len, figure_width))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(line_width)
plt.tick_params(width=line_width, length=tick_len)
# plt.yscale('symlog', linthreshy=0.1)

plt.plot(l_idx_peak_EI_STP_0, color='red', linewidth=plot_line_width, alpha=0.2)
plt.plot(l_idx_peak_EI_STP, color='red', linewidth=plot_line_width, alpha=0.4)
plt.plot(l_idx_peak_EI_STP_2, color='red', linewidth=plot_line_width, alpha=0.6)
plt.plot(l_idx_peak_EI_STP_3, color='red', linewidth=plot_line_width, alpha=1.0)
plt.plot(l_idx_ss_EI_STP_0, color='red', linestyle='dashed', linewidth=plot_line_width, alpha=0.2)
plt.plot(l_idx_ss_EI_STP, color='red', linestyle='dashed', linewidth=plot_line_width, alpha=0.4)
plt.plot(l_idx_ss_EI_STP_2, color='red', linestyle='dashed', linewidth=plot_line_width, alpha=0.6)
plt.plot(l_idx_ss_EI_STP_3, color='red', linestyle='dashed', linewidth=plot_line_width, alpha=1.0)


plt.xticks([0, 10, 20, 30, 40], [0, 0.25, 0.5, 0.75, 1.0], fontsize=font_size_1, **hfont)
plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0], fontsize=font_size_1, **hfont)
plt.xlabel('$p$', fontsize=font_size_1, **hfont)
plt.ylabel('Separation index', fontsize=font_size_1, **hfont)
plt.ylim([-1.05, 1.05])
plt.legend([r"$J_{EE}$: 0.0", r"$J_{EE}$: 1.1", r"$J_{EE}$: 1.2", r"$J_{EE}$: 1.3"], prop={"family": "Arial", 'size': font_size_1}, loc='lower right')
plt.savefig('paper_figures/png/Revision_Fig_Point_2_9_Morphing_EI_STP_changing_Jee_U_max_' + str(U_max) + '.png')
plt.savefig('paper_figures/pdf/Revision_Fig_Point_2_9_Morphing_EI_STP_changing_Jee_U_max_' + str(U_max) + '.pdf')


# plt.figure(figsize=(figure_len, figure_width))
# ax = plt.gca()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(True)
# ax.spines['left'].set_visible(True)
# for axis in ['top', 'bottom', 'left', 'right']:
#     ax.spines[axis].set_linewidth(line_width)
# plt.tick_params(width=line_width, length=tick_len)
#
# plt.yscale('symlog', linthreshy=0.1)
#
# plt.plot(l_peak_E1_EE_STP_0, color=pal[0], linewidth=plot_line_width, alpha=0.2)
# plt.plot(l_peak_E1_EE_STP, color=pal[0], linewidth=plot_line_width, alpha=0.4)
# plt.plot(l_peak_E1_EE_STP_2, color=pal[0], linewidth=plot_line_width, alpha=0.7)
# plt.plot(l_peak_E1_EE_STP_3, color=pal[0], linewidth=plot_line_width, alpha=1.0)
#
# plt.plot(l_peak_E2_EE_STP_0, color=pal[1], linewidth=plot_line_width, alpha=0.2)
# plt.plot(l_peak_E2_EE_STP, color=pal[1], linewidth=plot_line_width, alpha=0.4)
# plt.plot(l_peak_E2_EE_STP_2, color=pal[1], linewidth=plot_line_width, alpha=0.7)
# plt.plot(l_peak_E2_EE_STP_3, color=pal[1], linewidth=plot_line_width, alpha=1.0)
#
# plt.xticks([0, 10, 20, 30, 40], [0, 0.25, 0.5, 0.75, 1.0], fontsize=font_size_1, **hfont)
# plt.yticks([0, 0.1, 1, 10, 100, 1000], fontsize=font_size_1, **hfont)
# plt.ylabel('Firing rate (Hz)', fontsize=font_size_1, **hfont)
#
# plt.legend([r"$J_{EE}$: 0.0", r"$J_{EE}$: 1.1", r"$J_{EE}$: 1.2", r"$J_{EE}$: 1.3"], prop={"family": "Arial", 'size': font_size_1}, loc='lower right')
# plt.savefig('paper_figures/png/Fig_6_Morphing_EE_STP_activity_changing_Jee_U_max_' + str(U_max) + '_peak.png')
# plt.savefig('paper_figures/pdf/Fig_6_Morphing_EE_STP_activity_changing_Jee_U_max_' + str(U_max) + '_peak.pdf')
#
#
# plt.figure(figsize=(figure_len, figure_width))
# ax = plt.gca()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(True)
# ax.spines['left'].set_visible(True)
# for axis in ['top', 'bottom', 'left', 'right']:
#     ax.spines[axis].set_linewidth(line_width)
# plt.tick_params(width=line_width, length=tick_len)
#
# plt.yscale('symlog', linthreshy=0.1)
#
# plt.plot(l_ss_E1_EE_STP_0, color=pal[0], linewidth=plot_line_width, alpha=0.2)
# plt.plot(l_ss_E1_EE_STP, color=pal[0], linewidth=plot_line_width, alpha=0.4)
# plt.plot(l_ss_E1_EE_STP_2, color=pal[0], linewidth=plot_line_width, alpha=0.7)
# plt.plot(l_ss_E1_EE_STP_3, color=pal[0], linewidth=plot_line_width, alpha=1.0)
#
# plt.plot(l_ss_E2_EE_STP_0, color=pal[1], linewidth=plot_line_width, alpha=0.2)
# plt.plot(l_ss_E2_EE_STP, color=pal[1], linewidth=plot_line_width, alpha=0.4)
# plt.plot(l_ss_E2_EE_STP_2, color=pal[1], linewidth=plot_line_width, alpha=0.7)
# plt.plot(l_ss_E2_EE_STP_3, color=pal[1], linewidth=plot_line_width, alpha=1.0)
#
# plt.xticks([0, 10, 20, 30, 40], [0, 0.25, 0.5, 0.75, 1.0], fontsize=font_size_1, **hfont)
# plt.yticks([0, 0.1, 1, 10, 100, 1000], fontsize=font_size_1, **hfont)
# plt.ylabel('Firing rate (Hz)', fontsize=font_size_1, **hfont)
#
# plt.legend([r"$J_{EE}$: 0.0", r"$J_{EE}$: 1.1", r"$J_{EE}$: 1.2", r"$J_{EE}$: 1.3"], prop={"family": "Arial", 'size': font_size_1}, loc='lower right')
# plt.savefig('paper_figures/png/Fig_6_Morphing_EE_STP_activity_changing_Jee_U_max_' + str(U_max) + '_ss.png')
# plt.savefig('paper_figures/pdf/Fig_6_Morphing_EE_STP_activity_changing_Jee_U_max_' + str(U_max) + '_ss.pdf')