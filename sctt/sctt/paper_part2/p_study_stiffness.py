'''
Created on 11.05.2018

@author: Yingxiong
'''
import os.path

from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import numpy as np
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement \
    import ContinuousFibers
import scipy.stats as stats
from sctt.calibration.matrix_strength_dependence import interp_m_shape
from sctt.composite_tensile_test import CompositeTensileTest
from sctt.crack_bridge_models.random_bond_cb import RandomBondCB
from sctt.reinforcements.fiber_bundle import FiberBundle
from spirrid.rv import RV
from stats.misc.random_field.random_field_1D import RandomField
from stats.pdistrib.weibull_fibers_composite_distr import \
    WeibullFibers, fibers_MC
import traits.api as tr


home_dir = 'D:\\Eclipse\\'

ax1 = plt.subplot(122)
ax2 = plt.subplot(121)

load_exp_curves = False
if load_exp_curves:

    for i in range(5):
        path1 = [home_dir, 'git',  # the path of the data file
                 'rostar',
                 'scratch',
                 'diss_figs',
                 'TT-4C-0' + str(i + 1) + '.txt']
        filepath1 = filepath = os.path.join(*path1)

        path2 = [home_dir, 'git',  # the path of the data file
                 'rostar',
                 'scratch',
                 'diss_figs',
                 'TT-6C-0' + str(i + 1) + '.txt']
        filepath2 = os.path.join(*path2)

        plot_1percent = True
        if plot_1percent:
            data = np.loadtxt(filepath1, delimiter=';')
            ax1.plot(-data[:, 2] / 2. / 250. - data[:, 3] / 2. / 250.,
                     data[:, 1] / 2., lw=1, color='0.5')

        plot_15percent = True
        if plot_15percent:
            data = np.loadtxt(filepath2, delimiter=';')
            ax1.plot(-data[:, 2] / 2. / 250. - data[:, 3] / 2. / 250.,
                     data[:, 1] / 2., lw=1, color='0.5')

# simulation


class RVBeta(RV):
    a = tr.Float(0.05)
    b = tr.Float(0.6)
    loc = tr.Float(1e-4)
    scale = tr.Float(1.44)

    def __init__(self, type, loc=0.0, scale=0.0, shape=1.0,
                 *args, **kw):
        pass

    def ppf(self, x):
        return stats._continuous_distns.beta.ppf(x, self.a, self.b,
                                                 loc=self.loc, scale=self.scale)


rv_beta = RVBeta('beta')


def plot_eps_sig_cs(m_fiber, vf, mean, stdev):
    # specify the model parameters
    reinf = ContinuousFibers(r=3.5e-3,
                             #                              tau=RVBeta(
                             #                                  'beta'),
                             tau=RV(
                                 'gamma', loc=0.001260, scale=0.2 * 1.440, shape=0.0539),
                             V_f=vf,
                             E_f=182e3,
                             xi=fibers_MC(m=m_fiber, sV0=0.0076),
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
                               mean=mean,
                               stdev=stdev,
                               distr_type='Gauss')

    ctt.sig_mu_x = random_field.random_field

    sig_c_i, z_x_i, BC_x_i, sig_c_u, n_cracks = ctt.get_cracking_history()
    load_arr = np.linspace(0, sig_c_u, 100)
    print 'BC_x_i', BC_x_i
    eps_c_arr = ctt.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, load_arr)

    cs = 500. / (np.arange(len(z_x_i)) + 1)
    cs[cs > 120.] = 120.
    cs = np.hstack((cs, cs[-1]))

    eps_c_i = np.interp(np.hstack((sig_c_i, sig_c_u)), load_arr, eps_c_arr)

    return eps_c_arr, load_arr, eps_c_i, sig_c_i, cs


11
# the matrix strength parameters of the five speciemens - 1.0%
# m1_arr = np.array([2.8, 2.6, 2.8, 3.4, 3.3])
# s1_arr = np.array([0.10, 0.15, 0.15, 0.25, 0.08])

m1_arr = np.array([2.8])
s1_arr = np.array([0.10])

# the matrix strength parameters of the five speciemens - 1.5%
# m15_arr = np.array([2.8, 2.6, 2.8, 3.4, 3.3]) * 1.34
# s15_arr = np.array([0.10, 0.15, 0.15, 0.25, 0.08]) *1.34

# m15_arr = np.array([2.8]) * 1.34
# s15_arr = np.array([0.10]) * 1.34


# plot the curves
# eps_c_arr, load_arr, eps_c_i, cs = plot_eps_sig_cs(
#     4.0, 0.015, m15_arr, s15_arr)
# stress-strain curve
# ax1.plot(eps_c_arr, load_arr, lw=2, label='m_f = 4.0')
# crack development
# ax2.plot(eps_c_i, cs, drawstyle='steps', lw=2, label='m_f = 4.0')

eps_c_arr, load_arr, eps_c_i, sig_c_i, cs = plot_eps_sig_cs(
    6.7, 0.015, 2.8 * 1.34, 0.1 * 1.34)
# stress-strain curve
ax1.plot(eps_c_arr, load_arr, lw=2, label='1e-8*s_m')
# crack development
ax2.plot(eps_c_i, cs, drawstyle='steps', lw=2, label='1e-8*s_m')
ax1.plot(eps_c_i[0:-1], sig_c_i, '.')

ax1.plot([0., 0.007], [0., 0.007 * 1820 * 1.5], 'k--', lw=1)
plt.show()


eps_c_arr, load_arr, eps_c_i, sig_c_i, cs = plot_eps_sig_cs(
    6.7, 0.01, 2.8, 0.1)
# stress-strain curve
ax1.plot(eps_c_arr, load_arr, lw=2, label='1*s_m')
# crack development
ax2.plot(eps_c_i, cs, drawstyle='steps', lw=2, label='1*s_m')
ax1.plot(eps_c_i[0:-1], sig_c_i, '.')

# stiffness 180 GPa * 0.01
ax1.plot([0., 0.007], [0., 0.007 * 1950], 'k--', lw=1)
# stiffness 180 GPa * 0.015
ax1.plot([0., 0.007], [0., 0.007 * 1950 * 1.5], 'k--', lw=1)

# plt.legend(loc='best')
plt.show()


eps_c_arr, load_arr, eps_c_i, sig_c_i, cs = plot_eps_sig_cs(
    6.7, 0.015, 2.8 * 1.34, 0.01 * 1.34)
# stress-strain curve
ax1.plot(eps_c_arr, load_arr, lw=2, label='0.1*s_m')
# crack development
ax2.plot(eps_c_i, cs, drawstyle='steps', lw=2, label='0.1*s_m')
ax1.plot(eps_c_i[0:-1], sig_c_i, '.')

eps_c_arr, load_arr, eps_c_i, sig_c_i, cs = plot_eps_sig_cs(
    6.7, 0.015, 2.8 * 1.34, 0.05 * 1.34)
# stress-strain curve
ax1.plot(eps_c_arr, load_arr, lw=2, label='0.5*s_m')
# crack development
ax2.plot(eps_c_i, cs, drawstyle='steps', lw=2, label='0.5*s_m')
ax1.plot(eps_c_i[0:-1], sig_c_i, '.')


# plot the curves
eps_c_arr, load_arr, eps_c_i, sig_c_i, cs = plot_eps_sig_cs(
    6.7, 0.015, 2.8 * 1.34, 0.1 * 1.34)
# stress-strain curve
ax1.plot(eps_c_arr, load_arr, lw=2, label='1*s_m')
# crack development
ax2.plot(eps_c_i, cs, drawstyle='steps', lw=2, label='1*s_m')
ax1.plot(eps_c_i[0:-1], sig_c_i, '.')


eps_c_arr, load_arr, eps_c_i, sig_c_i, cs = plot_eps_sig_cs(
    6.7, 0.015, 2.8 * 1.34, 0.2 * 1.34)
# stress-strain curve
ax1.plot(eps_c_arr, load_arr, lw=2, label='2*s_m')
# crack development
ax2.plot(eps_c_i, cs, drawstyle='steps', lw=2, label='2*s_m')
ax1.plot(eps_c_i[0:-1], sig_c_i, '.')

# stiffness 180 GPa * 0.01
# ax1.plot([0., 0.007], [0., 0.007 * 1920], 'k--', lw=1)
# stiffness 180 GPa * 0.015
ax1.plot([0., 0.007], [0., 0.007 * 1820 * 1.5], 'k--', lw=1)

plt.legend(loc='best')
plt.show()


eps_c_arr, load_arr, eps_c_i, sig_c_i, cs = plot_eps_sig_cs(
    6.7, 0.01, 2.8, 1e-8)
# stress-strain curve
ax1.plot(eps_c_arr, load_arr, lw=2, label='0*s_m')
# crack development
ax2.plot(eps_c_i, cs, drawstyle='steps', lw=2, label='0*s_m')
ax1.plot(eps_c_i[0:-1], sig_c_i, '.')


eps_c_arr, load_arr, eps_c_i, sig_c_i, cs = plot_eps_sig_cs(
    6.7, 0.01, 2.8, 0.01)
# stress-strain curve
ax1.plot(eps_c_arr, load_arr, lw=2, label='0.1*s_m')
# crack development
ax2.plot(eps_c_i, cs, drawstyle='steps', lw=2, label='0.1*s_m')
ax1.plot(eps_c_i[0:-1], sig_c_i, '.')

eps_c_arr, load_arr, eps_c_i, sig_c_i, cs = plot_eps_sig_cs(
    6.7, 0.01, 2.8, 0.05)
# stress-strain curve
ax1.plot(eps_c_arr, load_arr, lw=2, label='0.5*s_m')
# crack development
ax2.plot(eps_c_i, cs, drawstyle='steps', lw=2, label='0.5*s_m')
ax1.plot(eps_c_i[0:-1], sig_c_i, '.')


# plot the curves
eps_c_arr, load_arr, eps_c_i, sig_c_i, cs = plot_eps_sig_cs(
    6.7, 0.01, 2.8, 0.1)
# stress-strain curve
ax1.plot(eps_c_arr, load_arr, lw=2, label='1*s_m')
# crack development
ax2.plot(eps_c_i, cs, drawstyle='steps', lw=2, label='1*s_m')
ax1.plot(eps_c_i[0:-1], sig_c_i, '.')


eps_c_arr, load_arr, eps_c_i, sig_c_i, cs = plot_eps_sig_cs(
    6.7, 0.01, 2.8, 0.2)
# stress-strain curve
ax1.plot(eps_c_arr, load_arr, lw=2, label='2*s_m')
# crack development
ax2.plot(eps_c_i, cs, drawstyle='steps', lw=2, label='2*s_m')
ax1.plot(eps_c_i[0:-1], sig_c_i, '.')


# from scipy.stats import linregress
# x = np.linspace(0.015, 0.025, 50)
# y = np.interp(x, eps_c_arr, load_arr)
# slope, intercept, r_value, p_value, std_err = linregress(x, y)
# print '1%', slope / 1.5


# plot the curves
# eps_c_arr, load_arr, eps_c_i, cs = plot_eps_sig_cs(
#     15., 0.015, m15_arr, s15_arr)
# stress-strain curve
# ax1.plot(eps_c_arr, load_arr, lw=2, label='m_f = 15')
# crack development
# ax2.plot(eps_c_i, cs, drawstyle='steps', lw=2, label='m_f = 15')

# plot the curves
# eps_c_arr, load_arr, eps_c_i, cs = plot_eps_sig_cs(
#     100., 0.015, m15_arr, s15_arr)
# stress-strain curve
# ax1.plot(eps_c_arr, load_arr, lw=2, label='m_f = 100')
# crack development
# ax2.plot(eps_c_i, cs, drawstyle='steps', lw=2, label='m_f = 100')


# stiffness 180 GPa * 0.01
ax1.plot([0., 0.007], [0., 0.007 * 1920], 'k--', lw=1)
# stiffness 180 GPa * 0.015
ax1.plot([0., 0.007], [0., 0.007 * 1920 * 1.5], 'k--', lw=1)

plt.legend(loc='best')
plt.show()
