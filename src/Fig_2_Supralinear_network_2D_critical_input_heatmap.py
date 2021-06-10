import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sympy.solvers import solve
from sympy import Symbol
from matplotlib import patches
import matplotlib.patches as mpatches
import scipy.io as sio
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap
from sympy.solvers import solve
from sympy import Symbol
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr
import scipy.io as sio

session = WolframLanguageSession()

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

# simulation setup
dt = 0.0001
T = int(5/dt)

# neuronal parameters
tau_e, tau_i = 0.020, 0.010
l_b_SSN = [True]

# network parameters
J_ie = 0.45
J_ei = 1.0
J_ii = 1.5

alpha_e, alpha_i = 2, 2
g_i = 2
tau_e, tau_i = 0.020, 0.010

pal = sns.color_palette()
color_list = pal.as_hex()

l_g_e = np.arange(1, 4+0.01, 0.01)
l_J_ee = np.arange(0.35, 0.60+0.001, 0.001)

fr_mat = np.zeros((len(l_J_ee), len(l_g_e))) * np.nan

for g_e_temp_idx in range(len(l_g_e)):
    g_e = l_g_e[g_e_temp_idx]

    for J_ee_idx in range(len(l_J_ee)):
        J_ee = l_J_ee[J_ee_idx]

        J = [[J_ee, -J_ei], [J_ie, -J_ii]]
        det_J = np.linalg.det(J)

        string = """ {solve_for} /. Solve[{{ (-rE + ({J_ee} * rE - {J_ei} * rI + {g_e}) ^ {alpha_e}) / {tau_e} == 0,
                          (-rI + ({J_ie} * rE - {J_ii} * rI + {g_i})^ {alpha_i}) / {tau_i} == 0}}, {solve_for}, Reals]""".format(
            J_ee=J_ee, J_ie=J_ie, J_ei=J_ei, J_ii=J_ii, alpha_e=alpha_e, alpha_i=alpha_i, g_e=g_e,
            g_i=g_i, tau_e=tau_e, tau_i=tau_i, solve_for='{rE, rI}')

        P = session.evaluate(string)

        if len(P) == 2:
            solution_string_1 = str(P[0])
            solution_rE_1 = solution_string_1.split(",")[0]
            if solution_rE_1 != 'Global`rE':

                solution_rI_1 = solution_string_1.split(",")[1]
                solution_string_2 = str(P[1])
                solution_rE_2 = solution_string_2.split(",")[0]
                solution_rI_2 = solution_string_2.split(",")[1]
                rE_1 = float(solution_rE_1[1:])
                rI_1 = float(solution_rI_1[:-1])
                rE_2 = float(solution_rE_2[1:])
                rI_2 = float(solution_rI_2[:-1])

                cond_1 = J_ee * rE_1 - J_ei * rI_1 + g_e
                cond_2 = J_ie * rE_1 - J_ii * rI_1 + g_i

                cond_3 = J_ee * rE_2 - J_ei * rI_2 + g_e
                cond_4 = J_ie * rE_2 - J_ii * rI_2 + g_i

                if (cond_1 >= 0) and (cond_2 >= 0):
                    fr_mat[J_ee_idx, g_e_temp_idx] = rE_1
                elif (cond_3 >= 0) and (cond_4 >= 0):
                    fr_mat[J_ee_idx, g_e_temp_idx] = rE_2
                else:
                    pass
            else:
                pass

        elif len(P) == 4:
            solution_string_1 = str(P[0])
            solution_rE_1 = solution_string_1.split(",")[0]
            solution_rI_1 = solution_string_1.split(",")[1]
            solution_string_2 = str(P[1])
            solution_rE_2 = solution_string_2.split(",")[0]
            solution_rI_2 = solution_string_2.split(",")[1]
            solution_string_3 = str(P[2])
            solution_rE_3 = solution_string_3.split(",")[0]
            solution_rI_3 = solution_string_3.split(",")[1]
            solution_string_4 = str(P[3])
            solution_rE_4 = solution_string_4.split(",")[0]
            solution_rI_4 = solution_string_4.split(",")[1]

            rE_1 = float(solution_rE_1[1:])
            rI_1 = float(solution_rI_1[:-1])
            rE_2 = float(solution_rE_2[1:])
            rI_2 = float(solution_rI_2[:-1])
            rE_3 = float(solution_rE_3[1:])
            rI_3 = float(solution_rI_3[:-1])
            rE_4 = float(solution_rE_4[1:])
            rI_4 = float(solution_rI_4[:-1])

            cond_1 = J_ee * rE_1 - J_ei * rI_1 + g_e
            cond_2 = J_ie * rE_1 - J_ii * rI_1 + g_i

            cond_3 = J_ee * rE_2 - J_ei * rI_2 + g_e
            cond_4 = J_ie * rE_2 - J_ii * rI_2 + g_i

            cond_5 = J_ee * rE_3 - J_ei * rI_3 + g_e
            cond_6 = J_ie * rE_3 - J_ii * rI_3 + g_i

            cond_7 = J_ee * rE_4 - J_ei * rI_4 + g_e
            cond_8 = J_ie * rE_4 - J_ii * rI_4 + g_i

            if (cond_1 >= 0) and (cond_2 >= 0):
                fr_mat[J_ee_idx, g_e_temp_idx] = rE_1
            elif (cond_3 >= 0) and (cond_4 >= 0):
                fr_mat[J_ee_idx, g_e_temp_idx] = rE_2
            else:
                pass

        else:
            pass

sio.savemat('data/Fig_2_Supralinear_network_2D_critical_input_heatmap.mat', mdict={'fr_mat': fr_mat})

plt.figure(figsize=(figure_len, figure_width))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(line_width)
plt.tick_params(width=line_width, length=tick_len)

g = sns.heatmap(fr_mat, cmap="bwr", norm=LogNorm(vmin=0.01, vmax=100), vmin=0.01, vmax=100)
g.set_facecolor('gray')
g.collections[0].colorbar.set_label("Hz")
g.figure.axes[-1].yaxis.label.set_size(font_size_1)
plt.xticks(np.arange(0.5, 301 + 100, 100), [1, 2, 3, 4], fontsize=font_size_1, **hfont)
plt.yticks(np.arange(0.5, 251 + 50, 50), [0.35, 0.4, 0.45, 0.5, 0.55, 0.60], fontsize=font_size_1, **hfont)
g.set_xticklabels(g.get_xticklabels(), rotation=0)
g.set_yticklabels(g.get_yticklabels(), rotation=0)
plt.xlim([0, 301])
plt.ylim([0, 251])
plt.xlabel(r'$g_E$', fontsize=font_size_1, **hfont)
plt.ylabel(r'$J_{EE}$', fontsize=font_size_1, **hfont)
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=font_size_1)
cbar = ax.collections[0].colorbar
cbar.set_ticks([0.01, 0.1, 1, 10, 100])
plt.savefig('paper_figures/png/Fig_2_Supralinear_network_2D_critical_input_heatmap.png')
plt.savefig('paper_figures/pdf/Fig_2_Supralinear_network_2D_critical_input_heatmap.png')