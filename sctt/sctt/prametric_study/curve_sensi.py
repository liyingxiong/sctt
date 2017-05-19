'''
Created on May 29, 2015

@author: Yingxiong
'''
import multiprocessing
from crack_bridge_models.random_bond_cb import RandomBondCB
import numpy as np
from scipy.interpolate import interp1d
import os.path
from reinforcements.fiber_bundle import FiberBundle
from stats.pdistrib.weibull_fibers_composite_distr import \
    WeibullFibers, fibers_MC
from composite_tensile_test import CompositeTensileTest
import matplotlib.pyplot as plt
from stats.misc.random_field.random_field_1D import RandomField
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement \
    import ContinuousFibers
from spirrid.rv import RV
from calibration.tau_strength_dependence import interp_tau_shape, interp_tau_scale
from calibration.matrix_strength_dependence import interp_m_shape
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.special import gamma

eps_max_lst = []
for j in range(5):
    filepath1 = 'D:\\data\\TT-4C-0' + str(j + 1) + '.txt'
    data = np.loadtxt(filepath1, delimiter=';')
    eps_max_lst.append(
        np.amax(-data[:, 2] / 2. / 250. - data[:, 3] / 2. / 250.))
eps_max = np.amin(eps_max_lst)

eps_arr = np.linspace(0, eps_max, 100)

sig_lst = []
for j in range(5):
    #     for j in [1]:
    filepath1 = 'D:\\data\\TT-4C-0' + str(j + 1) + '.txt'
    data = np.loadtxt(filepath1, delimiter=';')
    eps = -data[:, 2] / 2. / 250. - data[:, 3] / 2. / 250.
    sig = data[:, 1] / 2.
#     if j == 1:
#         interp_exp = interp1d(eps[0:1500],
#                               sig[0:1500], bounds_error=False, fill_value=0.)
#     else:
    interp_exp = interp1d(eps, sig, bounds_error=False, fill_value=0.)

    sig_lst.append(interp_exp(eps_arr))
#     plt.plot(eps_arr, interp_exp(eps_arr))

sig_avg = np.sum(sig_lst, axis=0) / 5.

plt.plot(eps_arr, sig_avg, 'w--', label='experiment')

cs = []
# for factor in [0.6, 0.8, 1.0, 1.2, 1.4]:
for factor in [1.4]:

    reinf = ContinuousFibers(r=3.5e-3,
                             tau=RV(
                                 'gamma', loc=0.001260, scale=1.440, shape=0.0539),
                             V_f=0.01,
                             E_f=180e3,
                             xi=fibers_MC(m=6.7, sV0=0.0076),
                             label='carbon',
                             n_int=500)
    cb = RandomBondCB(E_m=25e3,
                      reinforcement_lst=[reinf],
                      n_BC=10,
                      L_max=300)
    ctt = CompositeTensileTest(n_x=1000,
                               L=500.,
                               cb=cb)

    random_field = RandomField(seed=False,
                               lacor=1.,
                               length=500.,
                               nx=1000,
                               nsim=1,
                               loc=.0,
                               scale=3.1 * factor,
                               shape=60,
                               distr_type='Weibull')
    ctt.sig_mu_x = random_field.random_field

    sig_c_i, z_x_i, BC_x_i, sig_c_u, n_cracks = ctt.get_cracking_history()
    load_arr = np.linspace(0, sig_c_u, 100)
    eps_c_arr = ctt.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, load_arr)

    cs.append(500. / n_cracks)

    plt.plot(eps_c_arr, load_arr,
             label='$l_\\tau$=' + str(factor) + '$l_\\tau^\star$')
#              label='$s_\mathrm{m}$=' + str(factor) + '$s_\mathrm{m}^\star$')
    interp2 = interp1d(eps_c_arr, load_arr, bounds_error=False, fill_value=0.)
    sig2 = interp2(eps_arr)
    plt.fill_between(eps_arr, sig_avg, sig2, facecolor='blue')


# plt.legend(loc='best')

# plt.figure()
# plt.plot([0.6, 0.8, 1.0, 1.2, 1.4], cs)
# plt.xlabel('factor')
# plt.ylabel('crack spacing')
plt.xlabel('strain')
plt.ylabel('stress [Mpa]')
plt.show()
