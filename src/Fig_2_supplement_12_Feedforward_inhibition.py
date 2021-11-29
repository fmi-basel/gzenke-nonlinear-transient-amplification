import numpy as np
import scipy.io as sio

# simulation setup
dt = 0.0001
T = int(9/dt)

# neuronal parameters
tau_e, tau_i = 0.020, 0.010
alpha_e, alpha_i = 2, 2

# short-term depression
u_d = 1
tau_x = 0.20

# network connectivity
Jee = 1.0
Jei = 0.5
Jie = 1.2
Jii = 1.0

g_e_bs = 0.5
g_i_bs = 1.0

l_delta_g_e = np.arange(0, 3+0.01, 0.3)
l_delta_g_i = np.arange(0, 3+0.01, 0.3)
bs_mat, peak_mat, ss_mat = np.zeros((len(l_delta_g_i), len(l_delta_g_e))) * np.nan, np.zeros((len(l_delta_g_i), len(l_delta_g_e))) * np.nan, np.zeros((len(l_delta_g_i), len(l_delta_g_e))) * np.nan

l_b_no_STP = [False, True]

for b_no_STP in l_b_no_STP:

    for delta_g_e_idx in range(len(l_delta_g_e)):
        for delta_g_i_idx in range(len(l_delta_g_i)):

            delta_g_e = l_delta_g_e[delta_g_e_idx]
            delta_g_i = l_delta_g_i[delta_g_i_idx]

            g_e_stim = g_e_bs + delta_g_e
            g_i_stim = g_i_bs + delta_g_i

            r_e, r_i = 0, 0
            z_e, z_i = 0, 0
            x = 1

            l_r_e, l_r_i, l_x = [], [], []

            for i in range(T):
                if 50000 <= i < 70000:
                    g_e, g_i = g_e_stim, g_i_stim
                else:
                    g_e, g_i = g_e_bs, g_i_bs

                g_e = g_e * (g_e > 0)
                g_i = g_i * (g_i > 0)

                # SSN part
                if b_no_STP:
                    if 50000 <= i < 70000:
                        z_e = Jee * r_e - Jei * r_i + g_e
                        z_i = Jie * r_e - Jii * r_i + g_i
                    else:
                        z_e = Jee * x * r_e - Jei * r_i + g_e
                        z_i = Jie * r_e - Jii * r_i + g_i
                else:
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
            bs_mat[delta_g_i_idx, delta_g_e_idx] = np.mean(l_r_e[30000:40000])
            peak_mat[delta_g_i_idx, delta_g_e_idx] = np.max(l_r_e[50000:60000])
            ss_mat[delta_g_i_idx, delta_g_e_idx] = np.mean(l_r_e[60000:70000])

    if b_no_STP:
        sio.savemat('bs_mat_no_STP.mat', mdict={'bs_mat': bs_mat})
        sio.savemat('peak_mat_no_STP.mat', mdict={'peak_mat': peak_mat})
        sio.savemat('ss_mat_no_STP.mat', mdict={'ss_mat': ss_mat})
    else:
        sio.savemat('bs_mat.mat', mdict={'bs_mat': bs_mat})
        sio.savemat('peak_mat.mat', mdict={'peak_mat': peak_mat})
        sio.savemat('ss_mat.mat', mdict={'ss_mat': ss_mat})