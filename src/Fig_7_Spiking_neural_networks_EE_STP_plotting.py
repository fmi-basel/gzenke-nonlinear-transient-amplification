import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sympy.solvers import solve
from sympy import Symbol
from matplotlib import patches
import matplotlib.patches as mpatches
import scipy.io as sio
import random
from math import pi, sqrt, exp
import scipy.ndimage.filters as filters
import numpy.linalg as linalg

# plotting configuration
ratio = 1.5
figure_len, figure_width = 15*1.5, 12*1.5
font_size_1, font_size_2 = 36*ratio, 36*ratio
legend_size = 18*ratio
line_width, tick_len = 3*ratio, 10*ratio
marker_size = 15*ratio
marker_edge_width = 3 * ratio
plot_line_width = 5*ratio
hfont = {'fontname': 'Arial'}

sns.set(style='ticks')

# spiking raster plot
n_exc, n_inh = 800, 200
spike_mat_total = sio.loadmat('data/spiking_neural_network/Fig_7_Spiking_neural_networks_EE_STP_total_amp_150.mat')['spike_mat_total']

plt.figure(figsize=(18*3, 9*3))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(line_width)
plt.tick_params(width=line_width, length=tick_len)

for i in range(n_exc + n_inh):
    spike_bins = np.where(spike_mat_total[i, :] != 0)[0]
    if i < n_exc:
        plt.plot(spike_bins, np.ones(len(spike_bins)) * i, 'bo', markersize=2)  # excitatory neurons
    else:
        plt.plot(spike_bins, np.ones(len(spike_bins)) * i, 'ro', markersize=2)

plt.xticks(np.arange(0, 260000 + 10000, 130000), [0, 13, 26], fontsize=font_size_1, **hfont)
plt.yticks(np.arange(0, 1000 + 100, 200), fontsize=font_size_1, **hfont)
plt.xlabel('Time (s)', fontsize=font_size_1, **hfont)
plt.ylabel('Neurons', fontsize=font_size_1, **hfont)
plt.xlim([0, 260000])
plt.ylim([0, 1000])
plt.savefig('paper_figures/png/Fig_7_Spiking_neural_networks_EE_STP_raster_plot.png')



# filtered number of spikes
exc_spikes_0 = sio.loadmat('data/spiking_neural_network/Fig_7_Spiking_neural_networks_EE_STP_total_amp_150_0.mat')['exc_spikes_0'][0]
exc_spikes_1 = sio.loadmat('data/spiking_neural_network/Fig_7_Spiking_neural_networks_EE_STP_total_amp_150_1.mat')['exc_spikes_1'][0]
exc_spikes_2 = sio.loadmat('data/spiking_neural_network/Fig_7_Spiking_neural_networks_EE_STP_total_amp_150_2.mat')['exc_spikes_2'][0]
exc_spikes_3 = sio.loadmat('data/spiking_neural_network/Fig_7_Spiking_neural_networks_EE_STP_total_amp_150_3.mat')['exc_spikes_3'][0]
exc_spikes_4 = sio.loadmat('data/spiking_neural_network/Fig_7_Spiking_neural_networks_EE_STP_total_amp_150_4.mat')['exc_spikes_4'][0]

exc_spikes_4_idx = sio.loadmat('data/spiking_neural_network/Fig_7_Spiking_neural_networks_exc_4.mat')['exc'][0]
exc_spikes_4_idx_1 = exc_spikes_4_idx[:150]
exc_spikes_4_idx_2 = exc_spikes_4_idx[150:]

exc_spikes_4_1 = spike_mat_total[exc_spikes_4_idx_1, :]
exc_spikes_4_2 = spike_mat_total[exc_spikes_4_idx_2, :]

plt.figure(figsize=(figure_len, figure_width))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(line_width)
plt.tick_params(width=line_width, length=tick_len)

l_exc_spikes_fr_4_1, l_exc_spikes_fr_4_2 = [], []

for i in range(2600):
    l_exc_spikes_fr_4_1.append(np.sum(exc_spikes_4_1[:, i * 100:(i + 1) * 100]) / 150 * 100)
    l_exc_spikes_fr_4_2.append(np.sum(exc_spikes_4_2[:, i * 100:(i + 1) * 100]) / 50 * 100)

l_exc_spikes_fr_4_1_cut_1 = l_exc_spikes_fr_4_1[1600:2100]
l_exc_spikes_fr_4_2_cut_1 = l_exc_spikes_fr_4_2[1600:2100]
l_exc_spikes_fr_4_1_cut_2 = l_exc_spikes_fr_4_1[2100:]
l_exc_spikes_fr_4_2_cut_2 = l_exc_spikes_fr_4_2[2100:]

idx_4_1_1 = np.argmax(l_exc_spikes_fr_4_1_cut_1)
idx_4_2_1 = np.argmax(l_exc_spikes_fr_4_2_cut_1)
idx_4_1_2 = np.argmax(l_exc_spikes_fr_4_1_cut_2)
idx_4_2_2 = np.argmax(l_exc_spikes_fr_4_2_cut_2)


plt.plot(np.asarray(l_exc_spikes_fr_4_1), color='m', linewidth=plot_line_width)
plt.plot(np.asarray(l_exc_spikes_fr_4_2), color='gray', linewidth=plot_line_width)
plt.plot(idx_4_1_1+1600, np.max(l_exc_spikes_fr_4_1_cut_1), color='m', marker="^",
         markersize=marker_size * 4, markeredgewidth=marker_edge_width*2, markeredgecolor='black')  # transient 2
plt.plot(idx_4_2_1+1600, np.max(l_exc_spikes_fr_4_2_cut_1), color='gray', marker="^",
         markersize=marker_size * 4, markeredgewidth=marker_edge_width*2, markeredgecolor='black')  # transient 2
plt.plot(idx_4_1_2+2100, np.max(l_exc_spikes_fr_4_1_cut_2), color='m', marker="^",
         markersize=marker_size * 4, markeredgewidth=marker_edge_width*2, markeredgecolor='black')  # transient 2
plt.plot(idx_4_2_2+2100, np.max(l_exc_spikes_fr_4_2_cut_2), color='gray', marker="^",
         markersize=marker_size * 4, markeredgewidth=marker_edge_width*2, markeredgecolor='black')  # transient 2

plt.xticks(np.arange(1600, 2600 + 10, 200), [16, 18, 20, 22, 24, 26], fontsize=font_size_1, **hfont)
plt.yticks(np.arange(0, 300 + 10, 100), [0, 100, 200, 300], fontsize=font_size_1, **hfont)
plt.xlabel('Time (s)', fontsize=font_size_1, **hfont)
plt.ylabel('Firing rate (Hz)', fontsize=font_size_1, **hfont)
plt.xlim([1600, 2600])
plt.ylim([0, 300])
plt.legend(['Pattern 5 subset 1', 'Pattern 5 subset 2'], prop={"family": "Arial", 'size': font_size_1}, loc='upper right')

plt.savefig('paper_figures/png/Fig_7_Spiking_neural_networks_EE_STP_pattern_completion_overlap.png')
plt.savefig('paper_figures/pdf/Fig_7_Spiking_neural_networks_EE_STP_pattern_completion_overlap.pdf')


plt.figure(figsize=(figure_len, figure_width))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(line_width)
plt.tick_params(width=line_width, length=tick_len)

l_exc_spikes_fr_0, l_exc_spikes_fr_1, l_exc_spikes_fr_2, l_exc_spikes_fr_3, l_exc_spikes_fr_4 = [], [], [], [], []

for i in range(2200):
    l_exc_spikes_fr_0.append(np.sum(exc_spikes_0[i * 100:(i + 1) * 100]) / 200 * 100)
    l_exc_spikes_fr_1.append(np.sum(exc_spikes_1[i * 100:(i + 1) * 100]) / 200 * 100)
    l_exc_spikes_fr_2.append(np.sum(exc_spikes_2[i * 100:(i + 1) * 100]) / 200 * 100)
    l_exc_spikes_fr_3.append(np.sum(exc_spikes_3[i * 100:(i + 1) * 100]) / 200 * 100)
    l_exc_spikes_fr_4.append(np.sum(exc_spikes_4[i * 100:(i + 1) * 100]) / 200 * 100)

plt.plot(np.asarray(l_exc_spikes_fr_0), linewidth=plot_line_width)
plt.plot(np.asarray(l_exc_spikes_fr_1), linewidth=plot_line_width)
plt.plot(np.asarray(l_exc_spikes_fr_2), linewidth=plot_line_width)
plt.plot(np.asarray(l_exc_spikes_fr_3), linewidth=plot_line_width)
plt.plot(np.asarray(l_exc_spikes_fr_4), linewidth=plot_line_width)

plt.xticks(np.arange(0, 2200 + 10, 1100), [0, 11, 22], fontsize=font_size_1, **hfont)
plt.yticks(np.arange(0, 300 + 10, 100), [0, 100, 200, 300], fontsize=font_size_1, **hfont)
plt.xlabel('Time (s)', fontsize=font_size_1, **hfont)
plt.ylabel('Firing rate (Hz)', fontsize=font_size_1, **hfont)
plt.xlim([0, 2200])
plt.ylim([0, 300])
plt.legend(['Pattern 1', 'Pattern 2', 'Pattern 3', 'Pattern 4', 'Pattern 5'], prop={"family": "Arial", 'size': font_size_1}, loc='upper right')

plt.savefig('paper_figures/png/Fig_7_Spiking_neural_networks_EE_STP_overlap.png')
plt.savefig('paper_figures/pdf/Fig_7_Spiking_neural_networks_EE_STP_overlap.pdf')


n_dim = 2
n_neurons = 1000
bin_size = 100 # 10ms
n_bins = int(100000/bin_size)
partition_size = 5
data_binned = np.zeros((n_neurons, n_bins))  # 100ms bin

spike_mat_total_cut = spike_mat_total[:, :100000]
for i in range(n_neurons):
    for j in range(n_bins):
        data_binned[i, j] = np.sum(spike_mat_total_cut[i, j * bin_size:(j + 1) * bin_size])

data = np.zeros((n_neurons, n_bins))  # normalized data
data_mean = np.nanmean(data_binned, axis=1)
data_sd = np.std(data_binned, axis=1)

for i in range(data.shape[0]):
    data[i, :] = (data_binned[i, :] - data_mean[i]) / data_sd[i]

data_new_mean = np.nanmean(data, axis=1)
data_new_std = np.std(data, axis=1)

idx_total = set(np.arange(0, n_neurons, 1))
idx_set_exclude = set(np.where(data_sd == 0)[0])
idx_left = idx_total - idx_set_exclude

idx_left_array = list(idx_left)
idx_left_array = np.asarray(idx_left_array)

print(len(idx_left_array))

data_cut = data[idx_left_array, :]
print(data_cut.shape)

cov_mat = np.cov(data_cut)  # calculate the covariance matrix
print('Dim of the covariance matrix: ' + str(cov_mat.shape))

print(np.isnan(cov_mat).any())

eigenValues, eigenVectors = linalg.eig(cov_mat)  # calculate the eigenvalues and eigenvectors of the covariance matrix
idx = eigenValues.argsort()[::-1]  # sorted index in the original array
eigenValues = eigenValues[idx]  # sort the eigenvalues
print('Percentage of variance explained by the first two PC: ' + str(np.sum(eigenValues[:n_dim]) / np.sum(eigenValues)))

eigenVectors = eigenVectors[:, idx]  # sort the eigenvectors

PC = eigenVectors[:, :n_dim].T  # with transpose to make dim 3 * 20
print(PC.shape)

projection = np.matmul(PC, data_cut)
print(projection.shape)

centroid_1_x = np.mean(projection[0, int(n_bins / partition_size):int(n_bins / partition_size) * 2])
centroid_1_y = np.mean(projection[1, int(n_bins / partition_size):int(n_bins / partition_size) * 2])

print(centroid_1_x)
print(centroid_1_y)

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

l_bs_x_1, l_bs_y_1, l_bs_x_2, l_bs_y_2, l_bs_x_3, l_bs_y_3 = [], [], [], [], [], []
for i in range(len(projection[0, :int(n_bins / partition_size)])):
    l_bs_x_1.append(projection[0, i])
    l_bs_y_1.append(projection[1, i])

for i in range(len(projection[0, int(n_bins / partition_size) * 2:int(n_bins / partition_size) * 3])):
    l_bs_x_2.append(projection[0, int(n_bins / partition_size) * 2+i])
    l_bs_y_2.append(projection[1, int(n_bins / partition_size) * 2+i])

for i in range(len(projection[0, int(n_bins / partition_size) * 4:int(n_bins / partition_size) * 5])):
    l_bs_x_3.append(projection[0, int(n_bins / partition_size) * 4+i])
    l_bs_y_3.append(projection[1, int(n_bins / partition_size) * 4+i])

l_fp_x_1, l_fp_y_1, l_fp_x_2, l_fp_y_2 = [], [], [], []

for i in range(len(projection[0, int(n_bins / partition_size)+50:int(n_bins / partition_size)+150])):
    l_fp_x_1.append(projection[0, int(n_bins / partition_size)+50+i])
    l_fp_y_1.append(projection[1, int(n_bins / partition_size)+50+i])

for i in range(len(projection[0, int(n_bins / partition_size)*3+50:int(n_bins / partition_size)*3+150])):
    l_fp_x_2.append(projection[0, int(n_bins / partition_size)*3+50+i])
    l_fp_y_2.append(projection[1, int(n_bins / partition_size)*3+50+i])



plt.plot(np.mean(l_bs_x_1), np.mean(l_bs_y_1), color='black', marker='X', markersize=marker_size * 4, markeredgewidth=marker_edge_width, markeredgecolor='black')

plt.plot(np.mean(l_bs_x_2), np.mean(l_bs_y_2), color='black', marker='X', markersize=marker_size * 4, markeredgewidth=marker_edge_width, markeredgecolor='black')

plt.plot(np.mean(l_bs_x_3), np.mean(l_bs_y_3), color='black', marker='X', markersize=marker_size * 4, markeredgewidth=marker_edge_width, markeredgecolor='black')


plt.plot(projection[0, int(n_bins / partition_size) + 1:int(n_bins / partition_size) + 2],
         projection[1, int(n_bins / partition_size) + 1:int(n_bins / partition_size) + 2], color=pal[0], marker="^",
         markersize=marker_size * 4, markeredgewidth=marker_edge_width*2, markeredgecolor='black') # transient 1

plt.plot(projection[0, int(n_bins / partition_size) * 3 + 1:int(n_bins / partition_size) * 3 + 2],
         projection[1, int(n_bins / partition_size) * 3 + 1:int(n_bins / partition_size) * 3 + 2], color=pal[1], marker="^",
         markersize=marker_size * 4, markeredgewidth=marker_edge_width*2, markeredgecolor='black')  # transient 2

plt.plot(np.mean(l_fp_x_1), np.mean(l_fp_y_1), color=pal[0],
         marker='o', markersize=marker_size * 4, markeredgewidth=marker_edge_width*2, markeredgecolor='black')

plt.plot(np.mean(l_fp_x_2), np.mean(l_fp_y_2), color=pal[1],
         marker='o', markersize=marker_size * 4, markeredgewidth=marker_edge_width*2, markeredgecolor='black')

print(np.mean(l_fp_x_1))
print(np.mean(l_fp_y_1))
print(np.mean(l_fp_x_2))
print(np.mean(l_fp_y_2))

# plt.plot(projection[0, int(n_bins / partition_size) * 3 + 5:int(n_bins / partition_size) * 4],
#          projection[1, int(n_bins / partition_size) * 3 + 5:int(n_bins / partition_size) * 4], color=pal[1], marker='o',
#          markersize=marker_size * 4, markeredgewidth=marker_edge_width, markeredgecolor='black')

plt.plot(projection[0, :int(n_bins / partition_size * 2.5)], projection[1, :int(n_bins / partition_size * 2.5)],
         color=pal[0], linewidth=10)
plt.plot(projection[0, int(n_bins / partition_size * 2.5) :int(n_bins / partition_size) * 5],
         projection[1, int(n_bins / partition_size * 2.5) :int(n_bins / partition_size) * 5], color=pal[1], linewidth=10)


plt.xticks(fontsize=font_size_1, **hfont)
plt.yticks(fontsize=font_size_1, **hfont)
plt.xlabel('PC1', fontsize=font_size_1, **hfont)
plt.ylabel('PC2', fontsize=font_size_1, **hfont)
plt.xticks([-30, 0, 30, 60, 90], fontsize=font_size_1, **hfont)
plt.yticks([-80, -40, 0, 40, 80], fontsize=font_size_1, **hfont)
plt.xlim([-30, 90])
plt.ylim([-80, 80])

plt.savefig('paper_figures/png/Fig_7_Spiking_neural_networks_EE_STP_PCA.png')
plt.savefig('paper_figures/pdf/Fig_7_Spiking_neural_networks_EE_STP_PCA.pdf')

plt.figure(figsize=(figure_len, figure_width))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(line_width)
plt.tick_params(width=line_width, length=tick_len)

l_max_main, l_max_rest = [], []
l_average_main, l_average_rest = [], []

l_max_main.append(np.max(l_exc_spikes_fr_0[200:400]))
l_max_main.append(np.max(l_exc_spikes_fr_1[600:800]))
l_max_main.append(np.max(l_exc_spikes_fr_2[1000:1200]))
l_max_main.append(np.max(l_exc_spikes_fr_3[1400:1600]))
l_max_main.append(np.max(l_exc_spikes_fr_4[1800:2000]))

l_max_rest.append((np.max(l_exc_spikes_fr_1[200:400])+np.max(l_exc_spikes_fr_2[200:400])+np.max(l_exc_spikes_fr_3[200:400])+np.max(l_exc_spikes_fr_4[200:400]))/4)
l_max_rest.append((np.max(l_exc_spikes_fr_0[600:800]) +np.max(l_exc_spikes_fr_2[600:800])+np.max(l_exc_spikes_fr_3[600:800])+np.max(l_exc_spikes_fr_4[600:800]))/4)
l_max_rest.append((np.max(l_exc_spikes_fr_0[1000:1200])+np.max(l_exc_spikes_fr_1[1000:1200])+np.max(l_exc_spikes_fr_3[1000:1200])+np.max(l_exc_spikes_fr_4[1000:1200]))/4)
l_max_rest.append((np.max(l_exc_spikes_fr_0[1400:1600])+np.max(l_exc_spikes_fr_1[1400:1600])+np.max(l_exc_spikes_fr_2[1400:1600])+np.max(l_exc_spikes_fr_4[1400:1600]))/4)
l_max_rest.append((np.max(l_exc_spikes_fr_0[1800:2000])+np.max(l_exc_spikes_fr_1[1800:2000])+np.max(l_exc_spikes_fr_2[1800:2000])+np.max(l_exc_spikes_fr_3[1800:2000]))/4)

l_average_main.append(np.mean(l_exc_spikes_fr_0[250:350]))
l_average_main.append(np.mean(l_exc_spikes_fr_1[650:750]))
l_average_main.append(np.mean(l_exc_spikes_fr_2[1050:1150]))
l_average_main.append(np.mean(l_exc_spikes_fr_3[1450:1550]))
l_average_main.append(np.mean(l_exc_spikes_fr_4[1850:1950]))

l_average_rest.append((np.mean(l_exc_spikes_fr_1[250:350]) + np.mean(l_exc_spikes_fr_2[250:350]) + np.mean(
    l_exc_spikes_fr_3[250:350]) + np.mean(l_exc_spikes_fr_4[250:350])) / 4)
l_average_rest.append((np.mean(l_exc_spikes_fr_0[650:750]) + np.mean(l_exc_spikes_fr_2[650:750]) + np.mean(
    l_exc_spikes_fr_3[650:750]) + np.mean(l_exc_spikes_fr_4[650:750])) / 4)
l_average_rest.append((np.mean(l_exc_spikes_fr_0[1050:1150]) + np.mean(
    l_exc_spikes_fr_1[1050:1150]) + np.mean(l_exc_spikes_fr_3[1050:1150]) + np.mean(
    l_exc_spikes_fr_4[1050:1150])) / 4)
l_average_rest.append((np.mean(l_exc_spikes_fr_0[1450:1550]) + np.mean(
    l_exc_spikes_fr_1[1450:1550]) + np.mean(l_exc_spikes_fr_2[1450:1550]) + np.mean(
    l_exc_spikes_fr_4[1450:1550])) / 4)
l_average_rest.append((np.mean(l_exc_spikes_fr_0[1850:1950]) + np.mean(
    l_exc_spikes_fr_1[1850:1950]) + np.mean(l_exc_spikes_fr_2[1850:1950]) + np.mean(
    l_exc_spikes_fr_3[1850:1950])) / 4)

max_diff = np.asarray(l_max_main) - np.asarray(l_max_rest)
ss_diff = np.asarray(l_average_main) - np.asarray(l_average_rest)

ax = sns.barplot(data=[max_diff, ss_diff], linewidth=line_width, errwidth=line_width,
                 facecolor=(1, 1, 1, 0), errcolor=".2", edgecolor=".2")

for i in range(len(max_diff)):
    if i%2 == 0:
        plt.plot(0-0.1, max_diff[i], linestyle='none', marker='o', fillstyle='full',
                 markeredgewidth=marker_edge_width, markersize=marker_size * 1.5, markeredgecolor='black',
                 markerfacecolor='none')
    else:
        plt.plot(0+0.1, max_diff[i], linestyle='none', marker='o', fillstyle='full',
                 markeredgewidth=marker_edge_width, markersize=marker_size * 1.5, markeredgecolor='black',
                 markerfacecolor='none')

for i in range(len(ss_diff)):
    if i%2 == 0:
        plt.plot(1-0.1, ss_diff[i], linestyle='none', marker='o', fillstyle='full',
             markeredgewidth=marker_edge_width, markersize=marker_size*1.5, markeredgecolor='black',
             markerfacecolor='none')
    else:
        plt.plot(1+0.1, ss_diff[i], linestyle='none', marker='o', fillstyle='full',
             markeredgewidth=marker_edge_width, markersize=marker_size*1.5, markeredgecolor='black',
             markerfacecolor='none')


widthbars = [0.3, 0.3]
for bar, newwidth in zip(ax.patches, widthbars):
    x = bar.get_x()
    width = bar.get_width()
    centre = x + width / 2.
    bar.set_x(centre - newwidth / 2.)
    bar.set_width(newwidth)

plt.ylabel('Difference in firing rate (Hz)', fontsize=font_size_1)
plt.xticks(range(2), ['Peak amplitude', 'Fixed point'], fontsize=font_size_1)
plt.yticks([0, 100, 200], fontsize=font_size_1, **hfont)
plt.ylim([0, 200])
plt.savefig('paper_figures/png/Fig_7_Spiking_neural_networks_EE_STP_overlap_difference_in_firing_rate_barplot.png')
plt.savefig('paper_figures/pdf/Fig_7_Spiking_neural_networks_EE_STP_overlap_difference_in_firing_rate_barplot.pdf')
