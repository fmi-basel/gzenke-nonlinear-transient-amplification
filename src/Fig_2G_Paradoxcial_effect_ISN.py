import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl

# plotting configuration
ratio = 1.5
figure_len, figure_width = 15*ratio, 12*ratio
font_size_1, font_size_2 = 36*ratio, 36*ratio
legend_size = 18*ratio
line_width, tick_len = 3*ratio, 10*ratio
marker_size = 15*ratio
plot_line_width = 5*ratio
hfont = {'fontname': 'Arial'}

sns.set(style='ticks')
pal = sns.color_palette("deep")

tau_e = 0.020
tau_x = 0.200
U_d = 1
r_e = 1
l_x = np.arange(0, 1+0.01, 0.02)
l_J_ee = np.arange(0.0, 2+0.005, 0.01)
alpha_e = 2

l_x_pd, l_x_ISN_1, l_x_ISN_2, l_x_ISN = [], [], [], []
for J_ee in l_J_ee:
    l_x_pd.append(np.sqrt(1/(J_ee * alpha_e * np.power(r_e, (alpha_e-1)/alpha_e))))
    l_x_ISN_1.append(np.sqrt(1/(J_ee * alpha_e * np.power(r_e, (alpha_e-1)/alpha_e))))
    l_x_ISN_2.append((tau_x + tau_e + tau_e * tau_x * U_d * r_e)/(tau_x * J_ee * alpha_e * np.power(r_e, (alpha_e - 1)/alpha_e)))
    l_x_ISN.append(np.minimum(np.sqrt(1/(J_ee * alpha_e * np.power(r_e, (alpha_e-1)/alpha_e))), (tau_x + tau_e + tau_e * tau_x * U_d * r_e)/(tau_x * J_ee * alpha_e * np.power(r_e, (alpha_e - 1)/alpha_e))))

x_ISN_1 = l_x_ISN[1]
l_x_ISN[0] = x_ISN_1

plt.figure(figsize=(figure_len, figure_len))
ax = plt.gca()
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(line_width)
plt.tick_params(width=line_width, length=tick_len)

plt.plot(l_x_pd, linestyle='dashed', color='black', linewidth=plot_line_width)
plt.plot(l_x_ISN, linestyle='solid', color='black', linewidth=plot_line_width)
# plt.fill_between(np.arange(0, 200+0.5, 1), l_x_ISN, color=pal[0], alpha=0.6)
# plt.fill_between(np.arange(0, 200+0.5, 1), l_x_ISN, 1, color=pal[1], alpha=0.6)

# plt.fill_between(np.arange(0, 200+0.5, 1), l_x_pd, 1, color='none', hatch="X", edgecolor="gray", linewidth=0.0)

# cmap = ListedColormap(['#4c72b099', '#dd845299'])
# norm = mpl.colors.Normalize(vmin=-1, vmax=1)
# cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=[-0.5, 0.5])
# cbar.ax.set_yticklabels(['non-ISN', 'ISN'], fontsize=font_size_1, **hfont)  # vertically oriented colorbar

plt.xticks([0, 50, 100, 150, 200], [0, 0.5, 1.0, 1.5, 2.0], fontsize=font_size_1, **hfont)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=font_size_1, **hfont)
plt.xlabel(r'$J_{EE}$', fontsize=font_size_1, **hfont)
plt.ylabel('x', fontsize=font_size_1, **hfont)
plt.xlim([0, len(l_J_ee)-1])
plt.ylim([0, 1])

plt.savefig('paper_figures/png/Revision_Fig_Point_1_4_Paradoxical_effect_ISN.png')
plt.savefig('paper_figures/pdf/Revision_Fig_Point_1_4_Paradoxical_effect_ISN.pdf')


# # previous version
# plt.plot(l_x_pd, linestyle='dashed', color='black', linewidth=plot_line_width)
# plt.plot(l_x_ISN, linestyle='solid', color='black', linewidth=plot_line_width)
# plt.fill_between(np.arange(0, 200+0.5, 1), l_x_ISN, color=pal[0], alpha=0.6)
# plt.fill_between(np.arange(0, 200+0.5, 1), l_x_ISN, 1, color=pal[1], alpha=0.6)
#
# # plt.fill_between(np.arange(0, 200+0.5, 1), l_x_pd, 1, color='none', hatch="X", edgecolor="gray", linewidth=0.0)
#
# # cmap = ListedColormap(['#4c72b099', '#dd845299'])
# # norm = mpl.colors.Normalize(vmin=-1, vmax=1)
# # cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=[-0.5, 0.5])
# # cbar.ax.set_yticklabels(['non-ISN', 'ISN'], fontsize=font_size_1, **hfont)  # vertically oriented colorbar
#
# plt.xticks([0, 50, 100, 150, 200], [0, 0.5, 1.0, 1.5, 2.0], fontsize=font_size_1, **hfont)
# plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=font_size_1, **hfont)
# plt.xlabel(r'$J_{EE}$', fontsize=font_size_1, **hfont)
# plt.ylabel('x', fontsize=font_size_1, **hfont)
# plt.xlim([0, len(l_J_ee)-1])
# plt.ylim([0, 1])
#
# plt.savefig('paper_figures/png/Revision_Fig_Point_1_4_Paradoxical_effect_ISN.png')
# plt.savefig('paper_figures/pdf/Revision_Fig_Point_1_4_Paradoxical_effect_ISN.pdf')