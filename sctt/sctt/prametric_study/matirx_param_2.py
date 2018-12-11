'''
Created on May 19, 2015

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

for i, s_m in enumerate([2.82, 2.65, 2.85, 3.42, 3.30]):

    path1 = [home_dir, 'git',  # the path of the data file
             'rostar',
             'scratch',
             'diss_figs',
             'TT-4C-0' + str(i + 1) + '.txt']
    filepath1 = filepath = os.path.join(*path1)
    plt.cla()
    data = np.loadtxt(filepath1, delimiter=';')
    plt.plot(-data[:, 2] / 2. / 250. - data[:, 3] / 2. / 250.,
             data[:, 1] / 2., lw=1, color='0.5')

    random_field = RandomField(seed=False,
                               lacor=1.,
                               length=500.,
                               nx=1000,
                               nsim=1,
                               loc=.0,
                               scale=s_m,
                               shape=100.,
                               distr_type='Weibull')
    ctt.sig_mu_x = random_field.random_field

    sig_c_i, z_x_i, BC_x_i, sig_c_u, n_cracks = ctt.get_cracking_history()
    load_arr = np.linspace(0, sig_c_u, 100)
    eps_c_arr = ctt.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, load_arr)
    plt.plot(eps_c_arr, load_arr, 'k')
    path = 'D:\\1.0_' + str(i + 1) + '.pdf'
    plt.ylim((0, 25))
    plt.savefig(path, format='pdf')
    print(sig_c_u)
    print([sig_c_i])


reinf1 = ContinuousFibers(r=3.5e-3,
                          tau=RV(
                              'gamma', loc=0.001260, scale=1.440, shape=0.0539),
                          V_f=0.015,
                          E_f=180e3,
                          xi=fibers_MC(m=6.7, sV0=0.0076),
                          label='carbon',
                          n_int=500)

cb1 = RandomBondCB(E_m=25e3,
                   reinforcement_lst=[reinf1],
                   n_BC=10,
                   L_max=300)
ctt = CompositeTensileTest(n_x=1000,
                           L=500.,
                           cb=cb1)

for i, s_m in enumerate([4.70547783,  3.97960818,  4.36307537,  3.99053079,  4.14610819]):

    path2 = [home_dir, 'git',  # the path of the data file
             'rostar',
             'scratch',
             'diss_figs',
             'TT-6C-0' + str(i + 1) + '.txt']
    filepath2 = os.path.join(*path2)

    data = np.loadtxt(filepath2, delimiter=';')
    plt.cla()
    plt.plot(-data[:, 2] / 2. / 250. - data[:, 3] / 2. / 250.,
             data[:, 1] / 2., lw=1, color='0.5')

    random_field = RandomField(seed=False,
                               lacor=1.,
                               length=500.,
                               nx=1000,
                               nsim=1,
                               loc=.0,
                               scale=s_m,
                               shape=100.,
                               distr_type='Weibull')
    ctt.sig_mu_x = random_field.random_field

    sig_c_i, z_x_i, BC_x_i, sig_c_u, n_cracks = ctt.get_cracking_history()
    load_arr = np.linspace(0, sig_c_u, 100)
    eps_c_arr = ctt.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, load_arr)
    plt.plot(eps_c_arr, load_arr, 'k')
    path = 'D:\\1.5_' + str(i + 1) + '.pdf'
    plt.ylim((0, 25))
    plt.savefig(path, format='pdf')
    print(sig_c_u)
    print([sig_c_i])


plt.show()
