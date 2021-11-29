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
plot_line_width = 5*ratio
hfont = {'fontname': 'Arial'}

b_without_STP = True
b_EE_STD = True
b_EI_STF = True
b_weak_SFA = True
b_strong_SFA = True

if b_without_STP:
    # simulation setup
    dt = 0.0001
    T = int(9/dt)

    # neuronal parameters
    tau_e, tau_i = 0.020, 0.060
    alpha_e, alpha_i = 2, 2

    # network connectivity
    Jee = 0.9
    Jei = 0.5
    Jie = 1.2
    Jii = 0.5

    print(-Jee * Jii + Jei * Jie)

    r_e, r_i = 0, 0
    z_e, z_i = 0, 0

    l_r_e, l_r_i, l_x = [], [], []

    for i in range(T):
        if 50000 <= i < 70000:
            g_e, g_i = 2.0, 2
        else:
            g_e, g_i = 1.0, 2

        g_e = g_e * (g_e > 0)
        g_i = g_i * (g_i > 0)

        # SSN part
        z_e = Jee * r_e - Jei * r_i + g_e
        z_i = Jie * r_e - Jii * r_i + g_i

        z_e = z_e * (z_e > 0)
        z_i = z_i * (z_i > 0)

        r_e = r_e + (-r_e + np.power(z_e, alpha_e)) / tau_e * dt
        r_i = r_i + (-r_i + np.power(z_i, alpha_i)) / tau_i * dt

        r_e = r_e * (r_e > 0)
        r_i = r_i * (r_i > 0)

        l_r_e.append(r_e)
        l_r_i.append(r_i)

    l_r_e = np.asarray(l_r_e)
    l_r_i = np.asarray(l_r_i)

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
    plt.yscale('symlog', linthreshy=0.1)

    plt.plot(l_r_e, color='blue', linewidth=plot_line_width)
    plt.plot(l_r_i, color='red', linewidth=plot_line_width)

    plt.xticks(np.arange(30000, 90000 + 5000, 20000), np.arange(0, 6 + 0.5, 2), fontsize=font_size_1, **hfont)
    plt.yticks([0, 1, 100, 10000], fontsize=font_size_1, **hfont)
    plt.xlabel('Time (s)', fontsize=font_size_1, **hfont)
    plt.ylabel('Firing rate (Hz)', fontsize=font_size_1, **hfont)
    plt.xlim([30000, 90000])
    plt.ylim([0, 10000])
    plt.legend(['Exc', 'Inh'], prop={"family": "Arial", 'size': font_size_1}, loc='upper right')
    plt.savefig('paper_figures/png/Revision_Fig_Positive_detJ_without_STP.png')
    plt.savefig('paper_figures/pdf/Revision_Fig_Positive_detJ_without_STP.pdf')


if b_EE_STD:
    # simulation setup
    dt = 0.0001
    T = int(9 / dt)

    # neuronal parameters
    tau_e, tau_i = 0.020, 0.060
    alpha_e, alpha_i = 2, 2

    # short-term depression
    x, u_d = 1, 1
    tau_x = 0.20

    # network connectivity
    Jee = 0.9
    Jei = 0.5
    Jie = 1.2
    Jii = 0.5

    print(-Jee * Jii + Jei * Jie)

    r_e, r_i = 0, 0
    z_e, z_i = 0, 0

    l_r_e, l_r_i, l_x = [], [], []

    for i in range(T):
        if i == 40000:
            print(x)
            print(-Jee * x * Jii + Jei * Jie)

        if 50000 <= i < 70000:
            g_e, g_i = 2.0, 2
        else:
            g_e, g_i = 1.0, 2

        g_e = g_e * (g_e > 0)
        g_i = g_i * (g_i > 0)

        # SSN part
        z_e = Jee * x * r_e - Jei * r_i + g_e
        z_i = Jie * r_e - Jii * r_i + g_i

        z_e = z_e * (z_e > 0)
        z_i = z_i * (z_i > 0)

        r_e = r_e + (-r_e + np.power(z_e, alpha_e)) / tau_e * dt
        r_i = r_i + (-r_i + np.power(z_i, alpha_i)) / tau_i * dt

        r_e = r_e * (r_e > 0)
        r_i = r_i * (r_i > 0)

        x = x + ((1 - x) / tau_x - u_d * x * r_e) * dt
        x = np.clip(x, 0, 1)

        l_r_e.append(r_e)
        l_r_i.append(r_i)
        l_x.append(x)

    l_r_e = np.asarray(l_r_e)
    l_r_i = np.asarray(l_r_i)
    l_x = np.asarray(l_x)

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
    plt.yscale('symlog', linthreshy=0.1)

    plt.plot(l_r_e, color='blue', linewidth=plot_line_width)
    plt.plot(l_r_i, color='red', linewidth=plot_line_width)

    plt.xticks(np.arange(30000, 90000 + 5000, 20000), np.arange(0, 6 + 0.5, 2), fontsize=font_size_1, **hfont)
    plt.yticks([0, 1, 100, 10000], fontsize=font_size_1, **hfont)
    plt.xlabel('Time (s)', fontsize=font_size_1, **hfont)
    plt.ylabel('Firing rate (Hz)', fontsize=font_size_1, **hfont)
    plt.xlim([30000, 90000])
    plt.ylim([0, 10000])
    plt.legend(['Exc', 'Inh'], prop={"family": "Arial", 'size': font_size_1}, loc='upper right')

    plt.savefig('paper_figures/png/Revision_Fig_Positive_detJ_EE_STD.png')
    plt.savefig('paper_figures/pdf/Revision_Fig_Positive_detJ_EE_STD.pdf')


if b_EI_STF:
    # simulation setup
    dt = 0.0001
    T = int(9 / dt)

    # neuronal parameters
    tau_e, tau_i = 0.020, 0.060
    alpha_e, alpha_i = 2, 2

    # short-term facilitation
    U, U_max = 1, 6
    tau_x = 0.20
    x = 1

    # network connectivity
    Jee = 0.9
    Jei = 0.5
    Jie = 1.2
    Jii = 0.5

    print(-Jee * Jii + Jei * Jie)

    r_e, r_i = 0, 0
    z_e, z_i = 0, 0

    l_r_e, l_r_i, l_x = [], [], []

    for i in range(T):
        if i == 40000:
            print(x)
            print(-Jee * x * Jii + Jei * Jie)

        if 50000 <= i < 70000:
            g_e, g_i = 2.0, 2
        else:
            g_e, g_i = 1.0, 2

        g_e = g_e * (g_e > 0)
        g_i = g_i * (g_i > 0)

        # SSN part
        z_e = Jee * r_e - Jei * r_i + g_e
        z_i = Jie * x * r_e - Jii * r_i + g_i

        z_e = z_e * (z_e > 0)
        z_i = z_i * (z_i > 0)

        r_e = r_e + (-r_e + np.power(z_e, alpha_e)) / tau_e * dt
        r_i = r_i + (-r_i + np.power(z_i, alpha_i)) / tau_i * dt

        r_e = r_e * (r_e > 0)
        r_i = r_i * (r_i > 0)

        # adaptation of excitatory neurons
        x = x + ((U - x) / tau_x + U * (U_max - x) * r_e) * dt
        x = np.clip(x, 0, U_max)

        l_r_e.append(r_e)
        l_r_i.append(r_i)
        l_x.append(x)

    l_r_e = np.asarray(l_r_e)
    l_r_i = np.asarray(l_r_i)
    l_x = np.asarray(l_x)

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
    plt.yscale('symlog', linthreshy=0.1)

    plt.plot(l_r_e, color='blue', linewidth=plot_line_width)
    plt.plot(l_r_i, color='red', linewidth=plot_line_width)

    plt.xticks(np.arange(30000, 90000 + 5000, 20000), np.arange(0, 6 + 0.5, 2), fontsize=font_size_1, **hfont)
    plt.yticks([0, 1, 100, 10000], fontsize=font_size_1, **hfont)
    plt.xlabel('Time (s)', fontsize=font_size_1, **hfont)
    plt.ylabel('Firing rate (Hz)', fontsize=font_size_1, **hfont)
    plt.xlim([30000, 90000])
    plt.ylim([0, 10000])
    plt.legend(['Exc', 'Inh'], prop={"family": "Arial", 'size': font_size_1}, loc='upper right')

    plt.savefig('paper_figures/png/Revision_Fig_Positive_detJ_EI_STF.png')
    plt.savefig('paper_figures/pdf/Revision_Fig_Positive_detJ_EI_STF.pdf')


if b_weak_SFA:
    # simulation setup
    dt = 0.0001
    T = int(9 / dt)

    # neuronal parameters
    tau_e, tau_i = 0.020, 0.060
    alpha_e, alpha_i = 2, 2

    # adaptation
    a, b = 0, 1
    tau_a = 0.2

    # network connectivity
    Jee = 0.9
    Jei = 0.5
    Jie = 1.2
    Jii = 0.5

    r_e, r_i = 0, 0
    z_e, z_i = 0, 0

    l_r_e, l_r_i, l_a = [], [], []

    for i in range(T):

        if 50000 <= i < 70000:
            g_e, g_i = 2.0, 2
        else:
            g_e, g_i = 1.0, 2

        g_e = g_e * (g_e > 0)
        g_i = g_i * (g_i > 0)

        # SSN part
        z_e = Jee * r_e - Jei * r_i + g_e
        z_i = Jie * r_e - Jii * r_i + g_i

        z_e = z_e * (z_e > 0)
        z_i = z_i * (z_i > 0)

        r_e = r_e + (-r_e + np.power(z_e, alpha_e) - a) / tau_e * dt
        r_i = r_i + (-r_i + np.power(z_i, alpha_i)) / tau_i * dt

        r_e = r_e * (r_e > 0)
        r_i = r_i * (r_i > 0)

        # adaptation of excitatory neurons
        a = a + (-a + b * r_e) * dt / tau_a

        l_r_e.append(r_e)
        l_r_i.append(r_i)
        l_a.append(a)

    l_r_e = np.asarray(l_r_e)
    l_r_i = np.asarray(l_r_i)
    l_a = np.asarray(l_a)

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
    plt.yscale('symlog', linthreshy=0.1)

    plt.plot(l_r_e, color='blue', linewidth=plot_line_width)
    plt.plot(l_r_i, color='red', linewidth=plot_line_width)

    plt.xticks(np.arange(30000, 90000 + 5000, 20000), np.arange(0, 6 + 0.5, 2), fontsize=font_size_1, **hfont)
    plt.yticks([0, 1, 100, 10000], fontsize=font_size_1, **hfont)
    plt.xlabel('Time (s)', fontsize=font_size_1, **hfont)
    plt.ylabel('Firing rate (Hz)', fontsize=font_size_1, **hfont)
    plt.xlim([30000, 90000])
    plt.ylim([0, 10000])
    plt.legend(['Exc', 'Inh'], prop={"family": "Arial", 'size': font_size_1}, loc='upper right')

    plt.savefig('paper_figures/png/Revision_Fig_Positive_detJ_SFA_' + str(b) + '.png')
    plt.savefig('paper_figures/pdf/Revision_Fig_Positive_detJ_SFA_' + str(b) + '.pdf')


if b_strong_SFA:
    # simulation setup
    dt = 0.0001
    T = int(9 / dt)

    # neuronal parameters
    tau_e, tau_i = 0.020, 0.060
    alpha_e, alpha_i = 2, 2

    # adaptation
    a, b = 0, 10
    tau_a = 0.2

    # network connectivity
    Jee = 0.9
    Jei = 0.5
    Jie = 1.2
    Jii = 0.5

    r_e, r_i = 0, 0
    z_e, z_i = 0, 0

    l_r_e, l_r_i, l_a = [], [], []

    for i in range(T):

        if 50000 <= i < 70000:
            g_e, g_i = 2.0, 2
        else:
            g_e, g_i = 1.0, 2

        g_e = g_e * (g_e > 0)
        g_i = g_i * (g_i > 0)

        # SSN part
        z_e = Jee * r_e - Jei * r_i + g_e
        z_i = Jie * r_e - Jii * r_i + g_i

        z_e = z_e * (z_e > 0)
        z_i = z_i * (z_i > 0)

        r_e = r_e + (-r_e + np.power(z_e, alpha_e) - a) / tau_e * dt
        r_i = r_i + (-r_i + np.power(z_i, alpha_i)) / tau_i * dt

        r_e = r_e * (r_e > 0)
        r_i = r_i * (r_i > 0)

        # adaptation of excitatory neurons
        a = a + (-a + b * r_e) * dt / tau_a

        l_r_e.append(r_e)
        l_r_i.append(r_i)
        l_a.append(a)

    l_r_e = np.asarray(l_r_e)
    l_r_i = np.asarray(l_r_i)
    l_a = np.asarray(l_a)

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
    plt.yscale('symlog', linthreshy=0.1)

    plt.plot(l_r_e, color='blue', linewidth=plot_line_width)
    plt.plot(l_r_i, color='red', linewidth=plot_line_width)

    plt.xticks(np.arange(30000, 90000 + 5000, 20000), np.arange(0, 6 + 0.5, 2), fontsize=font_size_1, **hfont)
    plt.yticks([0, 1, 100, 10000], fontsize=font_size_1, **hfont)
    plt.xlabel('Time (s)', fontsize=font_size_1, **hfont)
    plt.ylabel('Firing rate (Hz)', fontsize=font_size_1, **hfont)
    plt.xlim([30000, 90000])
    plt.ylim([0, 10000])
    plt.legend(['Exc', 'Inh'], prop={"family": "Arial", 'size': font_size_1}, loc='upper right')

    plt.savefig('paper_figures/png/Revision_Fig_Positive_detJ_SFA_' + str(b) + '.png')
    plt.savefig('paper_figures/pdf/Revision_Fig_Positive_detJ_SFA_' + str(b) + '.pdf')