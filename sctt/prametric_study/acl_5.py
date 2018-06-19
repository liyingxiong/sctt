'''
Created on 16.04.2016

@author: Yingxiong
'''
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
import matplotlib.pyplot as plt

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

s_m_arr = np.linspace(2.5, 4.5, 5)
m_m_arr = np.linspace(20, 25, 5)

# fpath = 'D:\\data\\Tensile_test_multiple_cracking\\ss_curve_mm_24_1\\'

filepath1 = 'D:\\data\\Tensile_test_multiple_cracking\\TT-4C-05.txt'
data = np.loadtxt(filepath1, delimiter=';')
eps = -data[:, 2] / 2. / 250. - data[:, 3] / 2. / 250.
sig = data[:, 1] / 2.
plt.plot(eps, sig, lw=2)

eps_max_lst = []
for j in range(5):
    filepath1 = 'D:\\data\\Tensile_test_multiple_cracking\\ex\\TT-4C-0' + \
        str(j + 1) + '.txt'
    data = np.loadtxt(filepath1, delimiter=';')
    eps_max_lst.append(
        np.amax(-data[:, 2] / 2. / 250. - data[:, 3] / 2. / 250.))
eps_max = np.amin(eps_max_lst)
print eps_max
eps_arr = np.linspace(0, eps_max, 100)
interp_exp = interp1d(eps[0:1500],
                      sig[0:1500], bounds_error=False, fill_value=0.)
sig_arr = interp_exp(eps_arr)

# plt.plot(eps_arr, sig_arr)
# plt.show()


# for i in np.arange(5):
#
# print s_m
#
#     random_field = RandomField(seed=False,
#                                lacor=5.,
#                                length=500.,
#                                nx=1000,
#                                nsim=1,
#                                loc=.0,
#                                scale=s_m_arr[i],
#                                shape=m_m_arr[i],
#                                distr_type='Weibull')
#     ctt.sig_mu_x = random_field.random_field
#
#     sig_c_i, z_x_i, BC_x_i, sig_c_u, n_cracks = ctt.get_cracking_history()
#     load_arr = np.linspace(0, sig_c_u, 100)
#     eps_c_arr = ctt.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, load_arr)
#     plt.plot(eps_c_arr, load_arr, label=str(i))
#     interp_sim = interp1d(
#         eps_c_arr, load_arr, bounds_error=False, fill_value=0.)
#     lof1 = np.sum((interp_sim(eps_arr) - sig_arr) ** 2)
#     print lof1


plt.legend()
plt.show()


#     np.savetxt(fpath + str(i) + '.txt', np.vstack((load_arr, eps_c_arr)))

#     plt.plot(load_arr, eps_c_arr)
#     plt.show()
