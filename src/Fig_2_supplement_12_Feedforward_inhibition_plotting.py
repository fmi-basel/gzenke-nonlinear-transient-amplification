import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.patches as mpatches
import scipy.io as sio
from matplotlib.colors import LogNorm

# plotting configuration
ratio = 1.25
# plotting configuration
figure_len, figure_width = 15*1.5, 12*1.5
font_size_1, font_size_2 = 36*ratio, 36*ratio
legend_size = 18*ratio
line_width, tick_len = 3*ratio, 10*ratio
marker_size = 15*ratio
plot_line_width = 5*ratio
hfont = {'fontname': 'Arial'}
pal = sns.color_palette("deep")

sns.set(style='ticks')

l_b_no_STP = [False]

for b_no_STP in l_b_no_STP:
    if b_no_STP:
        bs_mat = sio.loadmat('data/bs_mat_no_STP.mat')['bs_mat']
        peak_mat = sio.loadmat('data/peak_mat_no_STP.mat')['peak_mat']
        ss_mat = sio.loadmat('data/ss_mat_no_STP.mat')['ss_mat']
    else:
        bs_mat = sio.loadmat('data/bs_mat.mat')['bs_mat']
        peak_mat = sio.loadmat('data/peak_mat.mat')['peak_mat']
        ss_mat = sio.loadmat('data/ss_mat.mat')['ss_mat']

    amp_mat = peak_mat/bs_mat
    amp_mat_ss = ss_mat/bs_mat

    plt.figure(figsize=(figure_len, figure_width))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)
    # g = sns.heatmap(amp_mat, cmap='RdBu_r', norm=LogNorm(amp_mat.min(), amp_mat.max()),
    #             vmin=0.0001, vmax=10000, linewidths=12)
    g = sns.heatmap(amp_mat, cmap='YlOrBr', norm=LogNorm(amp_mat.min(), amp_mat.max()),
                vmin=0.9, vmax=10000, linewidths=12)
    g.set_facecolor('gray')

    # ax.axhline(y=0, color='k',linewidth=10)
    # ax.axhline(y=amp_mat.shape[1], color='k',linewidth=12)
    # ax.axvline(x=0, color='k',linewidth=10)
    # ax.axvline(x=amp_mat.shape[0], color='k',linewidth=12)

    g.set_xticklabels(g.get_xticklabels(), rotation=0)
    g.set_yticklabels(g.get_yticklabels(), rotation=0)

    plt.xlabel(r"$\Delta g_E$", fontsize=font_size_1, **hfont)
    plt.ylabel(r"$\Delta g_I$", fontsize=font_size_1, **hfont)
    plt.xticks([], fontsize=font_size_1, **hfont)
    plt.yticks([], fontsize=font_size_1, **hfont)

    plt.xticks(np.arange(0.5, 16 + 1, 5), [0, 1.0, 2.0, 3.0], fontsize=font_size_1, **hfont)
    plt.yticks(np.arange(0.5, 16 + 1, 5), [0, 1.0, 2.0, 3.0], fontsize=font_size_1, **hfont)
    plt.xlim([0, 16])
    plt.ylim([0, 16])

    cax = plt.gcf().axes[-1]
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=font_size_1, axis='both', which='both', length=0)
    cbar.set_ticks([1, 1e2, 1e4])

    l_boundary = []
    for a_idx in np.arange(17):
        l_boundary.append(a_idx)

    l_boundary = np.asarray(l_boundary)
    ax2 = plt.twinx()
    sns.lineplot(data=l_boundary, color='black', linewidth=plot_line_width, ax=ax2)
    plt.yticks([])
    plt.ylim([0, 16])
    ax2.lines[0].set_linestyle("--")

    if b_no_STP:
        plt.savefig('paper_figures/png/Revision_Fig_Point_2_11_Feedforward_inhibition_no_STP.png')
        plt.savefig('paper_figures/pdf/Revision_Fig_Point_2_11_Feedforward_inhibition_no_STP.pdf')
    else:
        plt.savefig('paper_figures/png/Revision_Fig_Point_2_11_Feedforward_inhibition.png')
        plt.savefig('paper_figures/pdf/Revision_Fig_Point_2_11_Feedforward_inhibition.pdf')


    plt.figure(figsize=(figure_len, figure_width))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)
    g = sns.heatmap(amp_mat_ss, cmap='RdBu_r', norm=LogNorm(amp_mat_ss.min(), amp_mat_ss.max()), vmin=0.0001, vmax=10000, linewidths=12)
    g.set_facecolor('gray')

    g.set_xticklabels(g.get_xticklabels(), rotation=0)
    g.set_yticklabels(g.get_yticklabels(), rotation=0)

    plt.xlabel(r"$\Delta g_E$", fontsize=font_size_1, **hfont)
    plt.ylabel(r"$\Delta g_I$", fontsize=font_size_1, **hfont)
    plt.xticks([], fontsize=font_size_1, **hfont)
    plt.yticks([], fontsize=font_size_1, **hfont)

    plt.xticks(np.arange(0.5, 16 + 1, 5), [0, 1.0, 2.0, 3.0], fontsize=font_size_1, **hfont)
    plt.yticks(np.arange(0.5, 16 + 1, 5), [0, 1.0, 2.0, 3.0], fontsize=font_size_1, **hfont)
    plt.xlim([0, 16])
    plt.ylim([0, 16])

    cax = plt.gcf().axes[-1]
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=font_size_1, axis='both', which='both', length=0)
    cbar.set_ticks([1e-4, 1e-2, 1, 1e2, 1e4])

    l_boundary = []
    for a_idx in np.arange(17):
        l_boundary.append(a_idx)

    l_boundary = np.asarray(l_boundary)
    ax2 = plt.twinx()
    sns.lineplot(data=l_boundary, color='black', linewidth=plot_line_width, ax=ax2)
    plt.yticks([])
    plt.ylim([0, 16])
    ax2.lines[0].set_linestyle("--")

    if b_no_STP:
        plt.savefig('paper_figures/png/Revision_Fig_Point_2_11_Feedforward_inhibition_no_STP_ss.png')
        plt.savefig('paper_figures/pdf/Revision_Fig_Point_2_11_Feedforward_inhibition_no_STP_ss.pdf')
    else:
        plt.savefig('paper_figures/png/Revision_Fig_Point_2_11_Feedforward_inhibition_ss.png')
        plt.savefig('paper_figures/pdf/Revision_Fig_Point_2_11_Feedforward_inhibition_ss.pdf')


ratio = 1.5
figure_len, figure_width = 15*ratio, 12*ratio
font_size_1, font_size_2 = 36*ratio, 36*ratio
legend_size = 18*ratio
line_width, tick_len = 3*ratio, 10*ratio
marker_size = 15*ratio
marker_edge_width = 3 * ratio
plot_line_width = 5*ratio
hfont = {'fontname': 'Arial'}

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

l_ISN_index = np.ones(10)
l_ISN_index[6] = 100

for i in range(len(l_ISN_index)):
    plt.plot(i, l_ISN_index[i], linestyle='none', marker="*", fillstyle='full',
             markeredgewidth=marker_edge_width * 2, markersize=marker_size * 2, markeredgecolor='black',
             markerfacecolor='black')

plt.xticks([0, 2, 4, 6, 8, 10], [1.5, 1.6, 1.7, 1.8, 1.9, 2.0], fontsize=font_size_1, **hfont)
plt.yticks([-10000, -100, 0, 100, 10000], fontsize=font_size_1, **hfont)
plt.xlabel('$g_E$', fontsize=font_size_1, **hfont)
plt.ylabel('ISN index', fontsize=font_size_1, **hfont)
plt.xlim([-0.5, len(l_ISN_index)-0.5])
plt.ylim([-10000, 10000])
plt.hlines(y=0, xmin=0, xmax=len(l_ISN_index)-1, colors='k', linestyles=[(0, (6, 6, 6, 6))], linewidth=line_width)
plt.savefig('paper_figures/png/star_plot.png')
plt.savefig('paper_figures/pdf/star_plot.pdf')
