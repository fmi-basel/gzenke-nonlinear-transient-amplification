import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sympy.solvers import solve
from sympy import Symbol
from matplotlib import patches
import matplotlib.patches as mpatches
import scipy.io as sio

# plotting
figure_len, figure_width = 15, 12
font_size_1, font_size_2 = 36, 36
legend_size = 18
line_width, tick_len = 3, 10
marker_size = 15
plot_line_width = 5
hfont = {'fontname':'Helvetica'}

ratio = 1.5
# plotting configuration
figure_len, figure_width = 15*ratio, 12*ratio
font_size_1, font_size_2 = 36*ratio, 36*ratio
legend_size = 18*ratio
line_width, tick_len = 3*ratio, 10*ratio
marker_size = 15
plot_line_width = 5*ratio
hfont = {'fontname': 'Arial'}

sns.set(style='ticks')

# network parameters
n_exc, n_inh = 400, 100

spike_mat_total = np.zeros((n_exc+n_inh, 60000))
spike_mat_total_adaptation = np.zeros((n_exc+n_inh, 60000))
spike_mat_total_EE_STP = np.zeros((n_exc+n_inh, 60000))
spike_mat_total_EI_STP = np.zeros((n_exc+n_inh, 60000))

spike_mat_total[:, :10000] = sio.loadmat('data/spiking_neural_network/Fig_6S_Spiking_neural_networks_0.mat')['spike_mat']
spike_mat_total[:, 10000:20000] = sio.loadmat('data/spiking_neural_network/Fig_6S_Spiking_neural_networks_1.mat')['spike_mat']
spike_mat_total[:, 20000:30000] = sio.loadmat('data/spiking_neural_network/Fig_6S_Spiking_neural_networks_2.mat')['spike_mat']
spike_mat_total[:, 30000:40000] = sio.loadmat('data/spiking_neural_network/Fig_6S_Spiking_neural_networks_3.mat')['spike_mat']
spike_mat_total[:, 40000:50000] = sio.loadmat('data/spiking_neural_network/Fig_6S_Spiking_neural_networks_4.mat')['spike_mat']
spike_mat_total[:, 50000:] = sio.loadmat('data/spiking_neural_network/Fig_6S_Spiking_neural_networks_5.mat')['spike_mat']


spike_mat_total_adaptation[:, :10000] = sio.loadmat('data/spiking_neural_network/Fig_6S_Spiking_neural_networks_adaptation_0.mat')['spike_mat']
spike_mat_total_adaptation[:, 10000:20000] = sio.loadmat('data/spiking_neural_network/Fig_6S_Spiking_neural_networks_adaptation_1.mat')['spike_mat']
spike_mat_total_adaptation[:, 20000:30000] = sio.loadmat('data/spiking_neural_network/Fig_6S_Spiking_neural_networks_adaptation_2.mat')['spike_mat']
spike_mat_total_adaptation[:, 30000:40000] = sio.loadmat('data/spiking_neural_network/Fig_6S_Spiking_neural_networks_adaptation_3.mat')['spike_mat']
spike_mat_total_adaptation[:, 40000:50000] = sio.loadmat('data/spiking_neural_network/Fig_6S_Spiking_neural_networks_adaptation_4.mat')['spike_mat']
spike_mat_total_adaptation[:, 50000:] = sio.loadmat('data/spiking_neural_network/Fig_6S_Spiking_neural_networks_adaptation_5.mat')['spike_mat']


spike_mat_total_EE_STP[:, :10000] = sio.loadmat('data/spiking_neural_network/Fig_6S_Spiking_neural_networks_EE_STP_0.mat')['spike_mat']
spike_mat_total_EE_STP[:, 10000:20000] = sio.loadmat('data/spiking_neural_network/Fig_6S_Spiking_neural_networks_EE_STP_1.mat')['spike_mat']
spike_mat_total_EE_STP[:, 20000:30000] = sio.loadmat('data/spiking_neural_network/Fig_6S_Spiking_neural_networks_EE_STP_2.mat')['spike_mat']
spike_mat_total_EE_STP[:, 30000:40000] = sio.loadmat('data/spiking_neural_network/Fig_6S_Spiking_neural_networks_EE_STP_3.mat')['spike_mat']
spike_mat_total_EE_STP[:, 40000:50000] = sio.loadmat('data/spiking_neural_network/Fig_6S_Spiking_neural_networks_EE_STP_4.mat')['spike_mat']
spike_mat_total_EE_STP[:, 50000:] = sio.loadmat('data/spiking_neural_network/Fig_6S_Spiking_neural_networks_EE_STP_5.mat')['spike_mat']


spike_mat_total_EI_STP[:, :10000] = sio.loadmat('data/spiking_neural_network/Fig_6S_Spiking_neural_networks_EI_STP_0.mat')['spike_mat']
spike_mat_total_EI_STP[:, 10000:20000] = sio.loadmat('data/spiking_neural_network/Fig_6S_Spiking_neural_networks_EI_STP_1.mat')['spike_mat']
spike_mat_total_EI_STP[:, 20000:30000] = sio.loadmat('data/spiking_neural_network/Fig_6S_Spiking_neural_networks_EI_STP_2.mat')['spike_mat']
spike_mat_total_EI_STP[:, 30000:40000] = sio.loadmat('data/spiking_neural_network/Fig_6S_Spiking_neural_networks_EI_STP_3.mat')['spike_mat']
spike_mat_total_EI_STP[:, 40000:50000] = sio.loadmat('data/spiking_neural_network/Fig_6S_Spiking_neural_networks_EI_STP_4.mat')['spike_mat']
spike_mat_total_EI_STP[:, 50000:] = sio.loadmat('data/spiking_neural_network/Fig_6S_Spiking_neural_networks_EI_STP_5.mat')['spike_mat']



l_exc_fr, l_inh_fr = [], []
l_exc_fr_adaptation, l_inh_fr_adaptation = [], []
l_exc_fr_EE_STP, l_inh_fr_EE_STP = [], []
l_exc_fr_EI_STP, l_inh_fr_EI_STP = [], []
for i in range(600):
    l_exc_fr.append(np.sum(spike_mat_total[:400, i * 100:(i + 1) * 100]) / 400 * 100)
    l_inh_fr.append(np.sum(spike_mat_total[400:, i * 100:(i + 1) * 100]) / 100 * 100)

    l_exc_fr_adaptation.append(np.sum(spike_mat_total_adaptation[:400, i * 100:(i + 1) * 100]) / 400 * 100)
    l_inh_fr_adaptation.append(np.sum(spike_mat_total_adaptation[400:, i * 100:(i + 1) * 100]) / 100 * 100)

    l_exc_fr_EE_STP.append(np.sum(spike_mat_total_EE_STP[:400, i * 100:(i + 1) * 100]) / 400 * 100)
    l_inh_fr_EE_STP.append(np.sum(spike_mat_total_EE_STP[400:, i * 100:(i + 1) * 100]) / 100 * 100)

    l_exc_fr_EI_STP.append(np.sum(spike_mat_total_EI_STP[:400, i * 100:(i + 1) * 100]) / 400 * 100)
    l_inh_fr_EI_STP.append(np.sum(spike_mat_total_EI_STP[400:, i * 100:(i + 1) * 100]) / 100 * 100)


plt.figure(figsize=(figure_len, figure_width))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(line_width)
plt.tick_params(width=line_width, length=tick_len)


plt.plot(np.asarray(l_exc_fr), color='blue', linewidth=plot_line_width)
plt.plot(np.asarray(l_inh_fr), color='red', linewidth=plot_line_width)

plt.xticks(np.arange(0, 600 + 10, 200), [0, 2, 4, 6], fontsize=font_size_1, **hfont)
plt.yticks(np.arange(0, 400 + 10, 100), [0, 100, 200, 300, 400], fontsize=font_size_1, **hfont)
plt.xlabel('Time (s)', fontsize=font_size_1, **hfont)
plt.ylabel('Firing rate (Hz)', fontsize=font_size_1, **hfont)
plt.xlim([0, 600])
plt.ylim([0, 400])
plt.legend(['E', 'I'], prop={"family": "Arial", 'size': font_size_1}, loc='upper right')

plt.savefig('paper_figures/png/Fig_6S_Spiking_neural_networks_firing_rate.png')
plt.savefig('paper_figures/pdf/Fig_6S_Spiking_neural_networks_firing_rate.pdf')



plt.figure(figsize=(figure_len, figure_width))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(line_width)
plt.tick_params(width=line_width, length=tick_len)


plt.plot(np.asarray(l_exc_fr_adaptation), color='blue', linewidth=plot_line_width)
plt.plot(np.asarray(l_inh_fr_adaptation), color='red', linewidth=plot_line_width)

plt.xticks(np.arange(0, 600 + 10, 200), [0, 2, 4, 6], fontsize=font_size_1, **hfont)
plt.yticks(np.arange(0, 400 + 10, 100), [0, 100, 200, 300, 400], fontsize=font_size_1, **hfont)
plt.xlabel('Time (s)', fontsize=font_size_1, **hfont)
plt.ylabel('Firing rate (Hz)', fontsize=font_size_1, **hfont)
plt.xlim([0, 600])
plt.ylim([0, 400])
plt.legend(['E', 'I'], prop={"family": "Arial", 'size': font_size_1}, loc='upper right')

plt.savefig('paper_figures/png/Fig_6S_Spiking_neural_networks_adaptation_firing_rate.png')
plt.savefig('paper_figures/pdf/Fig_6S_Spiking_neural_networks_adaptation_firing_rate.pdf')



plt.figure(figsize=(figure_len, figure_width))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(line_width)
plt.tick_params(width=line_width, length=tick_len)


plt.plot(np.asarray(l_exc_fr_EE_STP), color='blue', linewidth=plot_line_width)
plt.plot(np.asarray(l_inh_fr_EE_STP), color='red', linewidth=plot_line_width)

plt.xticks(np.arange(0, 600 + 10, 200), [0, 2, 4, 6], fontsize=font_size_1, **hfont)
plt.yticks(np.arange(0, 400 + 10, 100), [0, 100, 200, 300, 400], fontsize=font_size_1, **hfont)
plt.xlabel('Time (s)', fontsize=font_size_1, **hfont)
plt.ylabel('Firing rate (Hz)', fontsize=font_size_1, **hfont)
plt.xlim([0, 600])
plt.ylim([0, 400])
plt.legend(['E', 'I'], prop={"family": "Arial", 'size': font_size_1}, loc='upper right')

plt.savefig('paper_figures/png/Fig_6S_Spiking_neural_networks_EE_STP_firing_rate.png')
plt.savefig('paper_figures/pdf/Fig_6S_Spiking_neural_networks_EE_STP_firing_rate.pdf')



plt.figure(figsize=(figure_len, figure_width))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(line_width)
plt.tick_params(width=line_width, length=tick_len)


plt.plot(np.asarray(l_exc_fr_EI_STP), color='blue', linewidth=plot_line_width)
plt.plot(np.asarray(l_inh_fr_EI_STP), color='red', linewidth=plot_line_width)

plt.xticks(np.arange(0, 600 + 10, 200), [0, 2, 4, 6], fontsize=font_size_1, **hfont)
plt.yticks(np.arange(0, 400 + 10, 100), [0, 100, 200, 300, 400], fontsize=font_size_1, **hfont)
plt.xlabel('Time (s)', fontsize=font_size_1, **hfont)
plt.ylabel('Firing rate (Hz)', fontsize=font_size_1, **hfont)
plt.xlim([0, 600])
plt.ylim([0, 400])
plt.legend(['E', 'I'], prop={"family": "Arial", 'size': font_size_1}, loc='upper right')

plt.savefig('paper_figures/png/Fig_6S_Spiking_neural_networks_EI_STP_firing_rate.png')
plt.savefig('paper_figures/pdf/Fig_6S_Spiking_neural_networks_EI_STP_firing_rate.pdf')