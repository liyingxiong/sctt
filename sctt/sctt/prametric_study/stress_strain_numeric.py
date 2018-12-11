'''
Created on Mar 22, 2015

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

# plot the experimental responses
home_dir = 'D:\\Eclipse\\'
for i in range(5):
    #     for i in [2]:
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

    data = np.loadtxt(filepath1, delimiter=';')
    plt.plot(-data[:, 2] / 2. / 250. - data[:, 3] / 2. / 250.,
             data[:, 1] / 2., lw=1, color='0.5')
    data = np.loadtxt(filepath2, delimiter=';')
    plt.plot(-data[:, 2] / 2. / 250. - data[:, 3] / 2. / 250.,
             data[:, 1] / 2., lw=1, color='0.5')

# matrix parameters
m_scale = 3.50
m_shape = interp_m_shape(m_scale)
print(m_shape)


# v_f=1.0%
reinf1 = ContinuousFibers(r=3.5e-3,
                          tau=RV(
                              'gamma', loc=0., scale=1.321, shape=0.066),
                          V_f=0.01,
                          E_f=180e3,
                          xi=fibers_MC(m=7.718, sV0=0.00996),
                          label='carbon',
                          n_int=500)

cb = RandomBondCB(E_m=25e3,
                  reinforcement_lst=[reinf1],
                  n_BC=12,
                  L_max=120.)

random_field = RandomField(seed=False,
                           lacor=1.,
                           length=120,
                           nx=400,
                           nsim=1,
                           loc=.0,
                           shape=m_shape,
                           scale=m_scale,
                           distr_type='Weibull')

ctt = CompositeTensileTest(n_x=400,
                           L=120,
                           cb=cb,
                           experimental_strength=13.5,
                           sig_mu_x=random_field.random_field)

sig_c_i, z_x_i, BC_x_i, sig_c_u = ctt.get_cracking_history()

print(('1.0%', [sig_c_i]))
print((np.sort(ctt.y)))


load_arr = np.unique(np.hstack((np.linspace(0, sig_c_u, 50), sig_c_i)))
eps_c_arr = ctt.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, load_arr)
#     damage_arr = ctt.get_damage_arr(sig_c_i, z_x_i, BC_x_i, load_arr)
crack_eps = ctt.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, sig_c_i)

plt.plot(eps_c_arr, load_arr, 'k', lw=2, label='$v_\mathrm{f}=1.0\%$')
plt.plot(crack_eps, sig_c_i, 'ko')

# v_f=1.5%
reinf2 = ContinuousFibers(r=3.5e-3,
                          tau=RV(
                              'gamma', loc=0.,  scale=1.321, shape=0.066),
                          V_f=0.015,
                          E_f=180e3,
                          xi=fibers_MC(m=7.718, sV0=0.00996),
                          label='carbon',
                          n_int=500)

cb1 = RandomBondCB(E_m=25e3,
                   reinforcement_lst=[reinf2],
                   n_BC=12,
                   L_max=120.)

random_field1 = RandomField(seed=False,
                            lacor=1.,
                            length=120,
                            nx=400,
                            nsim=1,
                            loc=.0,
                            shape=m_shape,
                            scale=1.218 * m_scale,
                            distr_type='Weibull')

ctt = CompositeTensileTest(n_x=400,
                           L=120,
                           cb=cb1,
                           experimental_strength=21.0,
                           sig_mu_x=random_field1.random_field)

sig_c_i, z_x_i, BC_x_i, sig_c_u = ctt.get_cracking_history()
eps_c_i = ctt.get_eps_c_i(sig_c_i, z_x_i, BC_x_i)

print(('1.5%', [sig_c_i]))
print((np.sort(ctt.y)))

load_arr = np.unique(np.hstack((np.linspace(0, sig_c_u, 50), sig_c_i)))
eps_c_arr = ctt.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, load_arr)
crack_eps = ctt.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, sig_c_i)

plt.plot(eps_c_arr, load_arr, 'k--', lw=2, label='$v_\mathrm{f}=1.5\%$')
plt.plot(crack_eps, sig_c_i, 'ko')

plt.legend(loc='best')
plt.show()
