import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sympy.solvers import solve
from sympy import Symbol
from matplotlib import patches
import matplotlib.patches as mpatches
import scipy.io as sio

# plotting configuration
ratio = 1.5
figure_len, figure_width = 15*ratio, 12*ratio
font_size_1, font_size_2 = 36*ratio, 36*ratio
legend_size = 18*ratio
line_width, tick_len = 3*ratio, 10*ratio
marker_size = 15
plot_line_width = 5*ratio
hfont = {'fontname': 'Arial'}

sns.set(style='ticks')

# simulation setup
dt = 0.0001
n_sec = 6
T = int(n_sec/dt)

# neuronal parameters
tau_e, tau_i = 0.020, 0.010

# network parameters
n_exc, n_inh = 400, 100

# network connectivity
J = np.zeros((n_exc+n_inh, n_exc+n_inh))
J[:n_exc, :n_exc] = 0.05
J[n_exc:, :n_exc] = 0.02
J[:n_exc, n_exc:] = 0.05
J[n_exc:, n_exc:] = 0.03

mask = np.random.choice([0, 1], size=(n_exc+n_inh, n_exc+n_inh), p=[0.8, 0.2])
J = np.multiply(J, mask)
np.fill_diagonal(J, 0)
sio.savemat('data/spiking_neural_network/Fig_6S_Spiking_neural_networks_connectivity_matrix.mat', mdict={'connectivity_matrix':J})

# membrane dynamics
v_exc = 0.000 # e reversal potential
v_rest = -0.070 # resting potential
v_inh = -0.080 # i reversal potential
tau_m_e = 0.020 # e membrane time constant
tau_m_i = 0.010 # i membrane time constant

# receptors
tau_ampa = 0.005 # time constant for ampa
tau_gaba = 0.010 # time constant for gaba
tau_nmda = 0.100 # time constant for nmda
alpha = 0.5 # weights for ampa and nmda, determine the overall exc conductance

# firing threshold
v_thr = np.ones(n_exc+n_inh) * (-0.050) # firing threshold
v_spike_rest = -0.07 # resetting membrane potential after spike
t_ref = 0.003 # refractory period
t_allow_spike = np.zeros(n_exc+n_inh) # allowed spike timing

# conductance based synapse
v = np.ones(n_exc+n_inh) * (-0.070) # membrane potential
g_inh = np.zeros(n_exc+n_inh) # inhibitory conductance
g_ampa = np.zeros(n_exc+n_inh) # ampa conductance
g_nmda = np.zeros(n_exc+n_inh) # nmda conductance
g_exc = np.zeros(n_exc+n_inh) # exc conductance

g_ext = np.zeros(n_exc+n_inh)
pre_spike_time = -100 * np.ones(n_exc+n_inh)  # last spike timing
t_allow_spike = np.zeros(n_exc+n_inh)  # last spike timing
b_last_spike = np.zeros(n_exc+n_inh)

spike_mat = np.zeros((n_exc+n_inh, 10000))
spike_mat_total = np.zeros((n_exc+n_inh, 60000))
fr = np.zeros(n_exc+n_inh)
firing_mat = np.zeros((n_exc+n_inh, n_sec))
idx = 0
fr_idx = 0

u_s = 0.2
tau_x = 0.20
x = np.ones((n_exc, n_exc))

J_EE = np.copy(J[:n_exc, :n_exc])

for i in range(T):
    t = dt * i
    t_prev = dt * (i - 1)
    if 20000 < i < 40000:
        g_e, g_i = 90, 30
    else:
        g_e, g_i = 30, 30

    g_ext[:n_exc] = np.random.random(n_exc) < (g_e*dt)
    g_ext[n_exc:] = np.random.random(n_inh) < (g_i*dt)

    # update recurrent input
    b_last_spike = (pre_spike_time == t_prev)  # boolean indicate whether neurons spiked at last time step

    # E-E short-term plasticty
    x = x - u_s * x * np.tile(b_last_spike[:n_exc], (n_exc, 1))
    x = x * (x >= 0)
    x = x * (x <= 1)
    x = x + (1-x)/tau_x * dt
    x = x * (x >= 0)
    x = x * (x <= 1)

    J[:n_exc, :n_exc] = J_EE * x

    exc_input = np.dot(J[:, :n_exc], b_last_spike[:n_exc])
    inh_input = np.dot(J[:, n_exc:], b_last_spike[n_exc:])
    exc_input = exc_input + g_ext

    # update conductance
    g_inh = g_inh + (-g_inh / float(tau_gaba)) * dt + inh_input
    g_ampa = g_ampa + (-g_ampa / float(tau_ampa)) * dt + exc_input
    g_nmda = g_nmda + ((-g_nmda + g_ampa) / float(tau_nmda)) * dt
    g_exc = alpha * g_ampa + (1 - alpha) * g_nmda

    # seperate inhibitory and excitatory
    v[:n_exc] = v[:n_exc] + ((v_rest - v[:n_exc]) + g_exc[:n_exc] * (v_exc - v[:n_exc]) + g_inh[:n_exc] * (v_inh - v[:n_exc])) / float(tau_m_e) * dt
    v[n_exc:] = v[n_exc:] + ((v_rest - v[n_exc:]) + g_exc[n_exc:] * (v_exc - v[n_exc:]) + g_inh[n_exc:] * (v_inh - v[n_exc:])) / float(tau_m_i) * dt

    spike_info = (v > v_thr) & (t > t_allow_spike)
    spike_neuron_idx = np.where(spike_info)[0]
    spike_neuron_exc_idx = set(np.arange(0, n_exc, 1)).intersection(spike_neuron_idx)
    spike_neuron_inh_idx = set(np.arange(n_exc, n_exc+n_inh, 1)).intersection(spike_neuron_idx)
    pre_spike_time[spike_neuron_idx] = t
    t_allow_spike[spike_neuron_idx] = t + t_ref
    v[spike_neuron_idx] = v_spike_rest  # reset membrane potential

    spike_mat[:, idx] = b_last_spike
    idx += 1
    if (i+1)%10000==0:
        fr_idx = int((i+1)/10000)-1
        plt.figure(figsize=(figure_len, figure_width))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)

        for i in range(n_exc+n_inh):
            spike_bins = np.where(spike_mat[i, :] != 0)[0]
            if i < n_exc:
                plt.plot(spike_bins, np.ones(len(spike_bins))*i, 'bo', markersize=5)    # excitatory neurons
            else:
                plt.plot(spike_bins, np.ones(len(spike_bins))*i, 'ro', markersize=5)

        spike_mat_total[:, fr_idx*10000:(fr_idx+1)*10000] = spike_mat
        fr = np.sum(spike_mat, 1)
        firing_mat[:, fr_idx] = np.copy(fr)
        plt.xticks(np.arange(0, 10000 + 2000, 2000), [0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=font_size_1, **hfont)
        plt.yticks(np.arange(0, 500+50, 100), fontsize=font_size_1, **hfont)
        plt.xlabel('Time (s)', fontsize=font_size_1, **hfont)
        plt.ylabel('Neurons', fontsize=font_size_1, **hfont)
        plt.xlim([0, 10000])
        plt.ylim([0, 500])
        plt.savefig('paper_figures/png/Fig_6S_Spiking_neural_networks_EE_STP_' + str(int(t)) + '.png')
        sio.savemat('data/spiking_neural_network/Fig_6S_Spiking_neural_networks_EE_STP_' + str(int(t)) + '.mat', mdict={'spike_mat': spike_mat})
        spike_mat = np.zeros((n_exc+n_inh, 10000))
        idx = 0

plt.figure(figsize=(figure_len, figure_width))
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

plt.xticks(np.arange(0, 60000 + 10000, 20000), [0, 2, 4, 6], fontsize=font_size_1, **hfont)
plt.yticks(np.arange(0, 500 + 50, 100), fontsize=font_size_1, **hfont)
plt.xlabel('Time (s)', fontsize=font_size_1, **hfont)
plt.ylabel('Neurons', fontsize=font_size_1, **hfont)
plt.xlim([0, 60000])
plt.ylim([0, 500])
plt.savefig('paper_figures/png/Fig_6S_Spiking_neural_networks_EE_STP_total.png')