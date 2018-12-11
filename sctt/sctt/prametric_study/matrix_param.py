'''
Created on Apr 23, 2015

@author: Yingxiong
'''
from crack_bridge_models.random_bond_cb import RandomBondCB
# from calibration import Calibration
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

home_dir = 'D:\\Eclipse\\'
path1 = [home_dir, 'git',  # the path of the data file
         'rostar',
         'scratch',
         'diss_figs',
         'TT-4C-01.txt']
filepath1 = os.path.join(*path1)
data = np.loadtxt(filepath1, delimiter=';')
eps_max = np.amax(-data[:, 2] / 2. / 250. - data[:, 3] / 2. / 250.)
eps_arr = np.linspace(0, eps_max, 100)
interp_exp = interp1d(-data[:, 2] / 2. / 250. - data[:, 3] / 2. / 250.,
                      data[:, 1] / 2., bounds_error=False, fill_value=0.)
sig_exp = interp_exp(eps_arr)

# plt.plot(eps_arr, sig_exp)
# plt.show()
# lack_of_fit = []
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

n = 15
s_m_arr = np.linspace(2.8, 3.8, n)
m_m_arr = np.linspace(10, 100, n)
X, Y = np.meshgrid(s_m_arr, m_m_arr)
lack_of_fit = np.zeros((n, n))
crack_spacing = np.zeros((n, n))

for i in range(n):
    for j in range(n):

        random_field = RandomField(seed=False,
                                   lacor=1.,
                                   length=500.,
                                   nx=1000,
                                   nsim=1,
                                   loc=.0,
                                   scale=X[i, j],
                                   shape=Y[i, j],
                                   distr_type='Weibull')
        ctt.sig_mu_x = random_field.random_field

        sig_c_i, z_x_i, BC_x_i, sig_c_u, n_cracks = ctt.get_cracking_history()
        load_arr = np.linspace(0, sig_c_u, 100)
        eps_c_arr = ctt.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, load_arr)
        interp_sim = interp1d(
            eps_c_arr, load_arr, bounds_error=False, fill_value=0.)
        sig_sim = interp_sim(eps_arr)
        lack_of_fit[i, j] = np.sum((sig_sim - sig_exp) ** 2)
        crack_spacing[i, j] = ctt.L / n_cracks

print([lack_of_fit])
print([crack_spacing])

plt.figure(figsize=(12, 9))
im = plt.imshow(lack_of_fit, interpolation='bilinear', origin='lower',
                cmap=cm.gray, extent=(2.8, 3.8, 10, 100), aspect=1. / 50.)
levels = np.arange(np.amin(lack_of_fit), np.amax(
    lack_of_fit), (np.amax(lack_of_fit) - np.amin(lack_of_fit)) / 10)
CS = plt.contour(lack_of_fit, levels,
                 origin='lower',
                 linewidths=1,
                 extent=(2.8, 3.8, 10, 100))
plt.clabel(CS, levels[1::2],  # label every second level
           inline=1,
           fmt='%1.1f',
           fontsize=12)
# CB = plt.colorbar(CS, shrink=0.8, extend='both')

plt.title('lack of fit-response curve')
plt.flag()

CBI = plt.colorbar(im, shrink=0.8)
plt.xlabel('s_m')
plt.ylabel('m_m')


def scale(shape):
    lp = 1.
    lc = 1000.
    sig_min = 2.72
    f = (lp / (lp + lc)) ** (1 / shape)
    return sig_min / (f * gamma(1 + 1 / shape))
s_m = scale(m_m_arr)
plt.plot(s_m, m_m_arr, '--')


lack_cs = np.abs(crack_spacing - 13.3) / 13.3

plt.figure(figsize=(12, 9))
im = plt.imshow(lack_cs, interpolation='bilinear', origin='lower',
                cmap=cm.gray, extent=(2.8, 3.8, 10, 100), aspect=1. / 50.)
levels = np.arange(np.amin(lack_cs), np.amax(
    lack_cs), (np.amax(lack_cs) - np.amin(lack_cs)) / 10)
CS = plt.contour(lack_of_fit, levels,
                 origin='lower',
                 linewidths=1,
                 extent=(2.8, 3.8, 10, 100))
plt.clabel(CS, levels[1::2],  # label every second level
           inline=1,
           fmt='%1.1f',
           fontsize=12)
# CB = plt.colorbar(CS, shrink=0.8, extend='both')

plt.title('lack of fit-crack spacing')
plt.flag()

CBI = plt.colorbar(im, shrink=0.8)
plt.xlabel('s_m')
plt.ylabel('m_m')
plt.plot(s_m, m_m_arr, '--')

plt.show()
