import matplotlib
matplotlib.use('Agg')

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# plotting configuration
ratio = 1.5
figure_len, figure_width = 15*ratio, 12*ratio
font_size_1, font_size_2 = 36*ratio, 36*ratio
legend_size = 18*ratio
line_width, tick_len = 3*ratio, 10*ratio
marker_size = 15*ratio
plot_line_width = 5*ratio


b_no_STP = False

for loop_idx in range(100):

    # simulation setup
    dt = 0.0001
    n_sec = 6
    T = int(n_sec / dt)

    # neuronal parameters
    tau_e, tau_i = 0.020, 0.010

    # network parameters
    n_exc, n_inh = 800, 200
    exc_assembly_size = 200
    inh_assembly_size = 50
    n_patterns = 5

    exc_random_pattern = np.zeros((n_patterns, exc_assembly_size), dtype=int)
    inh_random_pattern = np.zeros((n_patterns, inh_assembly_size), dtype=int)

    amp = 80

    # network connectivityq
    J = np.zeros((n_exc + n_inh, n_exc + n_inh))

    J_ee = 0.20
    J_ie = 0.09
    J_ei = 0.10
    J_ii = 0.10

    J_ee_2 = J_ee * 0.1
    J_ie_2 = J_ie * 0.6
    J_ei_2 = J_ei * 0.7
    J_ii_2 = J_ii * 0.1

    mask = np.random.choice([0, 1], size=(n_exc + n_inh, n_exc + n_inh), p=[0.8, 0.2])

    J[:n_exc, :n_exc] = J_ee_2
    J[n_exc:, :n_exc] = J_ie_2
    J[:n_exc, n_exc:] = J_ei_2
    J[n_exc:, n_exc:] = J_ii_2

    for i in range(n_patterns):
        exc_random_pattern_temp = np.sort(
            np.random.choice(np.arange(0, n_exc, 1), exc_assembly_size, replace=False))
        exc_random_pattern[i, :] = exc_random_pattern_temp.astype(int)
        inh_random_pattern_temp = np.sort(
            np.random.choice(np.arange(n_exc, n_exc + n_inh, 1), inh_assembly_size, replace=False))
        inh_random_pattern[i, :] = inh_random_pattern_temp.astype(int)

        sio.savemat(
            'data/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_exc_' + str(i) + '_amp_' + str(amp) + '_loop_idx_' + str(loop_idx) + '.mat',
            mdict={'exc': exc_random_pattern_temp})
        sio.savemat(
            'data/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_inh_' + str(i) + '_amp_' + str(amp) + '_loop_idx_' + str(loop_idx) + '.mat',
            mdict={'inh': exc_random_pattern_temp})

        for m in range(len(exc_random_pattern_temp)):
            for n in range(len(exc_random_pattern_temp)):
                exc_idx_1 = exc_random_pattern_temp[m]
                exc_idx_2 = exc_random_pattern_temp[n]
                J[exc_idx_1, exc_idx_2] = J_ee
                J[exc_idx_2, exc_idx_1] = J_ee

        for m in range(len(exc_random_pattern_temp)):
            for n in range(len(inh_random_pattern_temp)):
                exc_idx = exc_random_pattern_temp[m]
                inh_idx = inh_random_pattern_temp[n]
                J[inh_idx, exc_idx] = J_ie
                J[exc_idx, inh_idx] = J_ei

        for m in range(len(inh_random_pattern_temp)):
            for n in range(len(inh_random_pattern_temp)):
                inh_idx_1 = inh_random_pattern_temp[m]
                inh_idx_2 = inh_random_pattern_temp[n]
                J[inh_idx_1, inh_idx_2] = J_ii
                J[inh_idx_2, inh_idx_1] = J_ii

    J = np.multiply(J, mask)
    np.fill_diagonal(J, 0)

    # membrane dynamics
    v_exc = 0.000  # e reversal potential
    v_rest = -0.070  # resting potential
    v_inh = -0.080  # i reversal potential
    tau_m_e = 0.020  # e membrane time constant
    tau_m_i = 0.010  # i membrane time constant

    # receptors
    tau_ampa = 0.005  # time constant for ampa
    tau_gaba = 0.010  # time constant for gaba
    tau_nmda = 0.100  # time constant for nmda
    alpha = 0.5  # weights for ampa and nmda, determine the overall exc conductance

    # firing threshold
    v_thr = np.ones(n_exc + n_inh) * (-0.050)  # firing threshold
    v_spike_rest = -0.07  # resetting membrane potential after spike
    t_ref = 0.003  # refractory period
    t_allow_spike = np.zeros(n_exc + n_inh)  # allowed spike timing

    # conductance based synapse
    v = np.ones(n_exc + n_inh) * (-0.070)  # membrane potential
    g_inh = np.zeros(n_exc + n_inh)  # inhibitory conductance
    g_ampa = np.zeros(n_exc + n_inh)  # ampa conductance
    g_nmda = np.zeros(n_exc + n_inh)  # nmda conductance
    g_exc = np.zeros(n_exc + n_inh)  # exc conductance

    g_ext = np.zeros(n_exc + n_inh)
    pre_spike_time = -100 * np.ones(n_exc + n_inh)  # last spike timing
    t_allow_spike = np.zeros(n_exc + n_inh)  # last spike timing
    b_last_spike = np.zeros(n_exc + n_inh)

    spike_mat = np.zeros((n_exc + n_inh, 10000))
    spike_mat_total = np.zeros((n_exc + n_inh, int(n_sec / dt)))
    fr = np.zeros(n_exc + n_inh)
    firing_mat = np.zeros((n_exc + n_inh, n_sec))
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
            g_e = np.ones(n_exc) * 30
            g_e[exc_random_pattern[4, :150]] = amp
            # g_e = np.ones(n_exc) * 30
            # g_e[exc_random_pattern[0, :]] = amp

        elif 60000 < i < 80000:
            g_e = np.ones(n_exc) * 40
            g_e[exc_random_pattern[1, :]] = amp

        elif 100000 < i < 120000:
            g_e = np.ones(n_exc) * 30
            g_e[exc_random_pattern[2, :]] = amp

        elif 140000 < i < 160000:
            g_e = np.ones(n_exc) * 30
            g_e[exc_random_pattern[3, :]] = amp

        elif 180000 < i < 200000:
            g_e = np.ones(n_exc) * 30
            g_e[exc_random_pattern[4, :]] = amp

        elif 120000 < i < 240000:
            g_e = np.ones(n_exc) * 30
            g_e[exc_random_pattern[4, :150]] = amp

        else:
            g_e, g_i = np.ones(n_exc) * 30, np.ones(n_inh) * 30

        g_ext[:n_exc] = np.random.random(n_exc) < (g_e * dt)
        g_ext[n_exc:] = np.random.random(n_inh) < (g_i * dt)

        # update recurrent input
        b_last_spike = (pre_spike_time == t_prev)  # boolean indicate whether neurons spiked at last time step

        # E-E short-term plasticty
        x = x - u_s * x * np.tile(b_last_spike[:n_exc], (n_exc, 1))
        x = x * (x >= 0)
        x = x * (x <= 1)
        x = x + (1 - x) / tau_x * dt
        x = x * (x >= 0)
        x = x * (x <= 1)

        if b_no_STP:
            J[:n_exc, :n_exc] = J_EE
        else:
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
        v[:n_exc] = v[:n_exc] + ((v_rest - v[:n_exc]) + g_exc[:n_exc] * (v_exc - v[:n_exc]) + g_inh[:n_exc] * (
                    v_inh - v[:n_exc])) / float(tau_m_e) * dt
        v[n_exc:] = v[n_exc:] + ((v_rest - v[n_exc:]) + g_exc[n_exc:] * (v_exc - v[n_exc:]) + g_inh[n_exc:] * (
                    v_inh - v[n_exc:])) / float(tau_m_i) * dt

        spike_info = (v > v_thr) & (t > t_allow_spike)
        spike_neuron_idx = np.where(spike_info)[0]
        spike_neuron_exc_idx = set(np.arange(0, n_exc, 1)).intersection(spike_neuron_idx)
        spike_neuron_inh_idx = set(np.arange(n_exc, n_exc + n_inh, 1)).intersection(spike_neuron_idx)
        pre_spike_time[spike_neuron_idx] = t
        t_allow_spike[spike_neuron_idx] = t + t_ref
        v[spike_neuron_idx] = v_spike_rest  # reset membrane potential

        spike_mat[:, idx] = b_last_spike
        idx += 1
        if (i + 1) % 10000 == 0:
            fr_idx = int((i + 1) / 10000) - 1
            plt.figure(figsize=(figure_len, figure_width))
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(line_width)
            plt.tick_params(width=line_width, length=tick_len)

            exc_idx_0 = exc_random_pattern[0, :]
            exc_idx_1 = exc_random_pattern[1, :]
            exc_idx_2 = exc_random_pattern[2, :]
            exc_idx_3 = exc_random_pattern[3, :]
            exc_idx_4 = exc_random_pattern[4, :]

            exc_spikes_0 = np.sum(spike_mat[exc_idx_0, :], axis=0)
            exc_spikes_1 = np.sum(spike_mat[exc_idx_1, :], axis=0)
            exc_spikes_2 = np.sum(spike_mat[exc_idx_2, :], axis=0)
            exc_spikes_3 = np.sum(spike_mat[exc_idx_3, :], axis=0)
            exc_spikes_4 = np.sum(spike_mat[exc_idx_4, :], axis=0)

            plt.plot(exc_spikes_0, linewidth=plot_line_width)
            plt.plot(exc_spikes_1, linewidth=plot_line_width)
            plt.plot(exc_spikes_2, linewidth=plot_line_width)
            plt.plot(exc_spikes_3, linewidth=plot_line_width)
            plt.plot(exc_spikes_4, linewidth=plot_line_width)

            plt.xticks(np.arange(0, 10000 + 2000, 2000), [0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=font_size_1)
            plt.yticks([0, 2, 4, 6, 8, 10], fontsize=font_size_1)
            plt.xlabel('Time (s)', fontsize=font_size_1)
            plt.ylabel('Number of spikes', fontsize=font_size_1)
            plt.xlim([0, 10000])
            plt.ylim([0, 10])
            plt.legend(['Pattern 1', 'Pattern 2', 'Pattern 3', 'Pattern 4', 'Pattern 5'],
                       prop={"family": "Arial", 'size': font_size_1}, loc='upper right')

            if b_no_STP:
                plt.savefig('paper_figures/png/SNN/Revision_Fig_Point_1_2_Spiking_neural_networks_no_EE_STP_' + str(
                    int(t)) + 's_morphing_amp_' + str(amp) + '_loop_idx_' + str(loop_idx) + '.png')
            else:
                plt.savefig('paper_figures/png/SNN/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_' + str(
                    int(t)) + 's_amp_' + str(amp) + '_loop_idx_' + str(loop_idx) + '.png')

            plt.figure(figsize=(figure_len, figure_width))
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(line_width)
            plt.tick_params(width=line_width, length=tick_len)

            for j in range(n_exc + n_inh):
                spike_bins = np.where(spike_mat[j, :] != 0)[0]
                if j < n_exc:
                    plt.plot(spike_bins, np.ones(len(spike_bins)) * j, 'bo', markersize=5)  # excitatory neurons
                else:
                    plt.plot(spike_bins, np.ones(len(spike_bins)) * j, 'ro', markersize=5)

            spike_mat_total[:, fr_idx * 10000:(fr_idx + 1) * 10000] = spike_mat
            fr = np.sum(spike_mat, 1)
            firing_mat[:, fr_idx] = np.copy(fr)
            plt.xticks(np.arange(0, 10000 + 2000, 2000), [0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=font_size_1)
            plt.yticks(np.arange(0, 1000 + 100, 200), fontsize=font_size_1)
            plt.xlabel('Time (s)', fontsize=font_size_1)
            plt.ylabel('Neurons', fontsize=font_size_1)
            plt.xlim([0, 10000])
            plt.ylim([0, n_exc + n_inh])
            if b_no_STP:
                plt.savefig('paper_figures/png/SNN/Revision_Fig_Point_1_2_Spiking_neural_networks_no_EE_STP_' + str(
                    int(t)) + 's_morphing_amp_' + str(amp) + '_loop_idx_' + str(loop_idx) + '_spiking_plot.png')
            else:
                plt.savefig('paper_figures/png/SNN/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_' + str(
                    int(t)) + 's_amp_' + str(amp) + '_loop_idx_' + str(loop_idx) + '_spiking_plot.png')

            spike_mat = np.zeros((n_exc + n_inh, 10000))
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

    # plt.xticks(np.arange(0, 260000 + 10000, 130000), [0, 13, 26], fontsize=font_size_1, **hfont)
    plt.xticks(np.arange(0, 60000 + 10000, 20000), [0, 2, 4, 6], fontsize=font_size_1)
    plt.yticks(np.arange(0, 1000 + 100, 200), fontsize=font_size_1)
    plt.xlabel('Time (s)', fontsize=font_size_1)
    plt.ylabel('Neurons', fontsize=font_size_1)
    # plt.xlim([0, 260000])
    plt.xlim([0, 60000])
    plt.ylim([0, n_exc + n_inh])

    if b_no_STP:
        sio.savemat(
            'data/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_no_EE_STP_total_morphing_amp_' + str(
                amp) + '_loop_idx_' + str(loop_idx) + '.mat', mdict={'spike_mat_total': spike_mat_total})
        plt.savefig(
            'paper_figures/png/SNN/Revision_Fig_Point_1_2_Spiking_neural_networks_no_EE_STP_total_morphing_amp_' + str(
                amp) + '_loop_idx_' + str(loop_idx) + '.png')
    else:
        sio.savemat(
            'data/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_amp_' + str(
                amp) + '_loop_idx_' + str(loop_idx) + '.mat', mdict={'spike_mat_total': spike_mat_total})
        plt.savefig('paper_figures/png/SNN/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_amp_' + str(
            amp) + '_loop_idx_' + str(loop_idx) + '.png')

    plt.figure(figsize=(figure_len, figure_width))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)

    exc_idx_0 = exc_random_pattern[0, :]
    exc_idx_1 = exc_random_pattern[1, :]
    exc_idx_2 = exc_random_pattern[2, :]
    exc_idx_3 = exc_random_pattern[3, :]
    exc_idx_4 = exc_random_pattern[4, :]

    exc_spikes_0 = np.sum(spike_mat_total[exc_idx_0, :], axis=0)
    exc_spikes_1 = np.sum(spike_mat_total[exc_idx_1, :], axis=0)
    exc_spikes_2 = np.sum(spike_mat_total[exc_idx_2, :], axis=0)
    exc_spikes_3 = np.sum(spike_mat_total[exc_idx_3, :], axis=0)
    exc_spikes_4 = np.sum(spike_mat_total[exc_idx_4, :], axis=0)

    plt.plot(exc_spikes_0, linewidth=plot_line_width)
    plt.plot(exc_spikes_1, linewidth=plot_line_width)
    plt.plot(exc_spikes_2, linewidth=plot_line_width)
    plt.plot(exc_spikes_3, linewidth=plot_line_width)
    plt.plot(exc_spikes_4, linewidth=plot_line_width)

    plt.xticks(np.arange(0, 60000 + 10000, 20000), [0, 2, 4, 6], fontsize=font_size_1)
    plt.yticks([0, 2, 4, 6, 8, 10], fontsize=font_size_1)
    plt.xlabel('Time (s)', fontsize=font_size_1)
    plt.ylabel('Number of spikes', fontsize=font_size_1)
    plt.xlim([0, 60000])
    plt.ylim([0, 10])
    plt.legend(['Pattern 1', 'Pattern 2', 'Pattern 3', 'Pattern 4', 'Pattern 5'],
               prop={"family": "Arial", 'size': font_size_1}, loc='upper right')

    if b_no_STP:
        plt.savefig(
            'paper_figures/png/SNN/Revision_Fig_Point_1_2_Spiking_neural_networks_no_EE_STP_total_morphing_amp_' + str(
                amp) + '_loop_idx_' + str(loop_idx) + '_number_of_spikes.png')
    else:
        sio.savemat(
            'data/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_amp_' + str(
                amp) + '_loop_idx_' + str(loop_idx) + '_0.mat', mdict={'exc_spikes_0': exc_spikes_0})
        sio.savemat(
            'data/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_amp_' + str(
                amp) + '_loop_idx_' + str(loop_idx) + '_1.mat', mdict={'exc_spikes_1': exc_spikes_1})
        sio.savemat(
            'data/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_amp_' + str(
                amp) + '_loop_idx_' + str(loop_idx) + '_2.mat', mdict={'exc_spikes_2': exc_spikes_2})
        sio.savemat(
            'data/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_amp_' + str(
                amp) + '_loop_idx_' + str(loop_idx) + '_3.mat', mdict={'exc_spikes_3': exc_spikes_3})
        sio.savemat(
            'data/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_amp_' + str(
                amp) + '_loop_idx_' + str(loop_idx) + '_4.mat', mdict={'exc_spikes_4': exc_spikes_4})

        plt.savefig('paper_figures/png/SNN/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_amp_' + str(
            amp) + '_loop_idx_' + str(loop_idx) + '_number_of_spikes.png')

    # spiking raster plot
    spike_mat_total = sio.loadmat(
        'data/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_amp_' + str(
            amp) + '_loop_idx_' + str(loop_idx) + '.mat')['spike_mat_total']

    plt.figure(figsize=(18 * 3, 9 * 3))
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

    plt.xticks(np.arange(0, 60000 + 1000, 20000), [0, 2, 4, 6], fontsize=font_size_1)
    plt.yticks(np.arange(0, 1000 + 100, 200), fontsize=font_size_1)
    plt.xlabel('Time (s)', fontsize=font_size_1)
    plt.ylabel('Neurons', fontsize=font_size_1)
    plt.xlim([0, 60000])
    plt.ylim([0, 1000])
    plt.savefig('paper_figures/png/SNN/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_raster_plot_loop_idx_' + str(loop_idx) + '.png')


    exc_spikes_4_idx = sio.loadmat('data/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_exc_4_amp_' + str(amp) + '_loop_idx_' + str(loop_idx) + '.mat')['exc'][0]
    exc_spikes_4_idx_1 = exc_spikes_4_idx[:150]
    exc_spikes_4_idx_2 = exc_spikes_4_idx[150:]

    exc_spikes_4_1 = spike_mat_total[exc_spikes_4_idx_1, :]
    exc_spikes_4_2 = spike_mat_total[exc_spikes_4_idx_2, :]
    bl_firing_4_2 = np.sum(exc_spikes_4_2[:, 10000:20000], axis=1)
    ss_firing_4_2 = np.sum(exc_spikes_4_2[:, 30000:40000], axis=1)

    mean_bl_firing_4_2 = np.mean(bl_firing_4_2)
    mean_ss_firing_4_2 = np.mean(ss_firing_4_2)
    print(mean_bl_firing_4_2)
    print(mean_ss_firing_4_2)
    print(wilcoxon(bl_firing_4_2, ss_firing_4_2))

    sio.savemat('data_sum/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_mean_bl_firing_amp_' + str(amp) + '_' + str(loop_idx) + '.mat', mdict={'mean_bl_firing_4_2': mean_bl_firing_4_2})
    sio.savemat('data_sum/spiking_neural_network/Revision_Fig_Point_1_2_Spiking_neural_networks_EE_STP_total_mean_ss_firing_amp_' + str(amp) + '_' + str(loop_idx) + '.mat', mdict={'mean_ss_firing_4_2': mean_ss_firing_4_2})
