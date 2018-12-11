'''
Created on 22.11.2016

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
from calibration.matrix_strength_dependence import interp_m_shape

ax1 = plt.subplot(111)
ax2 = ax1.twinx()


# plot the experimental responses
home_dir = 'D:\\Eclipse\\'
# for i in range(5):
for i in [0]:
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

#     data = np.loadtxt(filepath1, delimiter=';')
#     ax1.plot(-data[:, 2] / 2. / 250. - data[:, 3] / 2. / 250.,
#              data[:, 1] / 2., lw=1, color='0.5')

    data = np.loadtxt(filepath2, delimiter=';')
    ax1.plot(-data[:, 2] / 2. / 250. - data[:, 3] / 2. / 250.,
             data[:, 1] / 2., lw=1, color='0.5')

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
mean_arr = np.array([2.8, 2.6, 2.8, 3.4, 3.3])
stdev_arr = np.array([0.10, 0.15, 0.15, 0.25, 0.08])

# mean_arr = np.array([2.8])
# stdev_arr = np.array([0.1])
#
#
for m, s in zip(mean_arr, stdev_arr):

    random_field = RandomField(seed=False,
                               lacor=1.,
                               length=500.,
                               nx=1000,
                               nsim=1,
                               mean=m,
                               stdev=s,
                               distr_type='Gauss')

    ctt.sig_mu_x = random_field.random_field

    sig_c_i, z_x_i, BC_x_i, sig_c_u, n_cracks = ctt.get_cracking_history()
    load_arr = np.linspace(0, sig_c_u, 100)
    eps_c_arr = ctt.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, load_arr)
    ax1.plot(eps_c_arr, load_arr, lw=2)

    cs = 500. / (np.arange(len(z_x_i)) + 1)
    cs[cs > 120.] = 120.
    cs = np.hstack((cs, cs[-1]))

    eps_c_i = np.interp(np.hstack((sig_c_i, sig_c_u)), load_arr, eps_c_arr)
    ax2.plot(eps_c_i, cs, drawstyle='steps', lw=2)

    print(('cs', cs[-1]))
    print('====================')

# =========================================================================
# 1% aramis
# s1t = np.array([10.00872464,   5.52844059,  11.25707195,
#                 7.12078782,   7.5985656,   7.12078782,  12.27597275])
# s2t = np.array([17.1316769,   9.72990228,  6.97622649,   8.94871857])
# s3t = np.array(
#     [4.40479296,   7.95527934,  17.34835118,   8.85553078,   6.56429542])
# s4t = np.array(
#     [15.18919693,   7.0975642,   7.97860545,  11.93039465,   9.31819916])
# s5t = np.array([21.13306692,   8.62724404,  18.52434985,
#                 14.0797884,   7.63826555,   8.38268409,  21.98885654])
#
# s1b = np.array(
#     [6.02457869, 22.90243875,  11.25707195,   7.35146734,   8.93051279,  16.18415792])
# s2b = np.array(
#     [7.10445464,   9.2492387,   6.97622649,  16.79998289, 8.77070183,   8.77070183])
# s3b = np.array(
#     [4.40479296,  14.90546437,   5.49085621,   4.40479296, 5.97395657,  17.01455883])
# s4b = np.array([19.68039936,   6.88058764,  13.92738198,
#                 7.0975642, 7.0975642,   7.97860545,  17.20897856])
# s5b = np.array([22.92756902,   8.62724404,   8.99976055,
#                 10.78376281, 7.17470687,  12.65419438,   8.23534141,   5.11059319])
# strength_e = [12.5, 10.38, 11.34, 12.43, 13.40]
# =========================================================================

#=========================================================================
#1.5% aramis

s1t = np.array([16.91681074,  28.24358356,  14.16653556,   7.85852962,
                9.26719295,  35.72474096,  15.07669087,  11.44987103,
                9.26719295,   9.89696334,   9.58805621])
s2t = np.array([14.03091356,  18.79005001,  13.87798694,  15.37755049,
                17.00879121,  15.98687316,  15.59358517,  12.76183435])
s3t = np.array([9.05175008,  10.76772668,   9.05175008,   9.05175008,
                19.71997199,  11.42490661,  11.21500315,  12.15511804,
                7.663163,  18.42709064,   6.37574066,  15.91778204])
s4t = np.array([10.36301934,  18.73350336,  10.00397938,   7.78785911,
                11.22337517,  10.72917466,  11.89792408,  17.01613892,  10.86517866])
s5t = np.array([18.0952497,    0.60585937,
                19.90342057,   0.60585937,  31.91502337, 11.28946654,  11.71460843,   0.60585937,  12.16635002])

s1b = np.array([11.85898609,  19.53944724,
                11.44987103,  11.71533759,  14.91634299,  34.56317101,
                10.66047702,   7.48661655,  40.40901103])
s2b = np.array([10.48396872,  25.14811366,  12.08379738,  11.80306615,
                11.3517029,  10.27731759,  14.15291021,  32.14101299,  10.74935872])

s3b = np.array([16.02677067,   7.47010424,   8.7946856,  13.85198624,
                7.663163,   4.19339853,  26.06350928,   7.47010424,
                8.25237909,   7.25361537])
s4b = np.array([16.51271353,  13.22650031,  37.56411316,   9.14751061,
                32.29994302,  18.90466892,   7.54397453,   9.34653494])

s5b = np.array([10.81934181,   0.60585937,  35.98725918,  10.65285666,   0.60585937,   0.6058362,
                0.60549606,  15.01184154,   0.60585937])
strength_e = [20.66, 21.29, 20.76, 20.63, 19.60]
#=========================================================================

s1e = np.sort(np.hstack((s1t, s1b)))
s2e = np.sort(np.hstack((s2t, s2b)))
s3e = np.sort(np.hstack((s3t, s3b)))
s4e = np.sort(np.hstack((s4t, s4b)))
s5e = np.sort(np.hstack((s5t, s5b)))
s_lste = [s1e, s2e, s3e, s4e, s5e]

# for i in range(4):

for i in [0]:
    path1 = [home_dir, 'git',  # the path of the data file
             'rostar',
             'scratch',
             'diss_figs',
             'TT-6C-0' + str(i + 1) + '.txt']
    filepath1 = filepath = os.path.join(*path1)

    data = np.loadtxt(filepath1, delimiter=';')
    experi_eps, experi_sig = -data[:, 2] / 2. / \
        250. - data[:, 3] / 2. / 250., data[:, 1] / 2.

    a = s_lste[i]
    nc = np.arange(len(a))
    cs = 240. / (nc + 1)
    cs[cs >= 120] = 120

    strain = np.interp(
        np.hstack((0, a / 2, strength_e[i])), experi_sig, experi_eps)
    ax2.step(strain, np.hstack((120, cs, cs[-1])), color='0.5')
    ax2.set_ylim(0,)


plt.show()
