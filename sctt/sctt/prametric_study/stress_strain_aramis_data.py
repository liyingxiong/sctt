'''
Created on Mar 22, 2015

@author: Yingxiong
'''
import numpy as np
import os
import matplotlib.pyplot as plt
from crack_bridge_models.random_bond_cb import RandomBondCB
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement \
    import ContinuousFibers
from spirrid.rv import RV
from stats.pdistrib.weibull_fibers_composite_distr import \
    WeibullFibers, fibers_MC
from sctt_aramis import CTTAramis

#========================================================
# draw the experimental responses
#========================================================
home_dir = 'D:\\Eclipse\\'

path1 = [home_dir, 'git',  # the path of the data file
         'rostar',
         'scratch',
         'diss_figs',
         'TT-4C-05.txt']
filepath1 = filepath = os.path.join(*path1)

path2 = [home_dir, 'git',  # the path of the data file
         'rostar',
         'scratch',
         'diss_figs',
         'TT-6C-02.txt']
filepath2 = os.path.join(*path2)

data = np.loadtxt(filepath1, delimiter=';')
plt.plot(-data[:, 2] / 2. / 250. - data[:, 3] / 2. / 250.,
         data[:, 1] / 2., lw=1, color='0.5')
data = np.loadtxt(filepath2, delimiter=';')
plt.plot(-data[:, 2] / 2. / 250. - data[:, 3] / 2. / 250.,
         data[:, 1] / 2., lw=1, color='0.5')

# 4 layers
reinf1 = ContinuousFibers(r=3.5e-3,
                          tau=RV(
                              'gamma', loc=0., scale=2.276, shape=0.0505),
                          V_f=0.01,
                          E_f=180e3,
                          xi=fibers_MC(m=8.806, sV0=0.0134),
                          label='carbon',
                          n_int=500)

cb = RandomBondCB(E_m=25e3,
                  reinforcement_lst=[reinf1],
                  n_BC=12,
                  L_max=120.)
force1 = np.array([7.3054, 8.3827, 8.4770, 9.1553, 9.1553,
                   11.9988, 14.9728, 15.8779, 20.9879, 20.9879])
position1 = np.array([66.9834, 106.4972, 22.4680, 40.2742,
                      83.7893, 29.9706, 113.4996, 57.9803, 8.1630, 94.3930])

ctta = CTTAramis(n_x=400,
                 L=120,
                 cb=cb,
                 stress=force1 / 2.,
                 position=position1)

sig_c_i, z_x_i, BC_x_i = ctta.gen_data()

load_arr = np.unique(np.hstack((np.linspace(0, 13.5, 50), sig_c_i)))
eps_c_arr = ctta.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, load_arr)
crack_eps_a = ctta.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, sig_c_i)

plt.plot(eps_c_arr, load_arr, 'k', lw=2, label='vf=1.0')
plt.plot(crack_eps_a, sig_c_i, 'ko')

# six layers
reinf2 = ContinuousFibers(r=3.5e-3,
                          tau=RV(
                              'gamma', loc=0., scale=2.276, shape=0.0505),
                          V_f=0.015,
                          E_f=180e3,
                          xi=fibers_MC(m=8.806, sV0=0.0134),
                          label='carbon',
                          n_int=500)

cb = RandomBondCB(E_m=25e3,
                  reinforcement_lst=[reinf2],
                  n_BC=12,
                  L_max=120.)
force1 = np.array([9.7460, 10.3671, 10.6549, 10.7494, 11.4416,
                   13.4239, 14.3986, 14.5634, 15.1167, 16.8489, 22.0983])
position1 = np.array([49.2052, 60.6131, 31.8932, 103.2426,
                      24.7883, 55.5095, 78.2253, 41.2997, 90.8340, 69.4192, 11.7793])

ctta = CTTAramis(n_x=400,
                 L=120,
                 cb=cb,
                 stress=force1 / 2.,
                 position=position1)

sig_c_i, z_x_i, BC_x_i = ctta.gen_data()

load_arr = np.unique(np.hstack((np.linspace(0, 21.0, 50), sig_c_i)))
eps_c_arr = ctta.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, load_arr)
crack_eps_a = ctta.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, sig_c_i)

plt.plot(eps_c_arr, load_arr, 'k--', lw=2, label='vf=1.5')
plt.plot(crack_eps_a, sig_c_i, 'ko')

plt.show()
