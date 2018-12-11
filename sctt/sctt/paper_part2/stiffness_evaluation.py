'''
Created on 16.05.2018

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
from scipy.stats import linregress

home_dir = 'D:\\Eclipse\\'

x = np.linspace(0.002, 0.005, 100)

vf1 = []
vf15 = []

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
        y = np.interp(x, -data[:, 2] / 2. / 250. - data[:, 3] / 2. / 250.,
                      data[:, 1] / 2.)
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        print(('1%', slope))
        vf1.append(slope)
        plt.plot(-data[:, 2] / 2. / 250. - data[:, 3] / 2. / 250.,
                 data[:, 1] / 2.)

    plot_15percent = True
    if plot_15percent:
        data = np.loadtxt(filepath2, delimiter=';')
        y = np.interp(x, -data[:, 2] / 2. / 250. - data[:, 3] / 2. / 250.,
                      data[:, 1] / 2.)
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        print(('1.5%', slope / 1.5))
        vf15.append(slope)
        plt.plot(-data[:, 2] / 2. / 250. - data[:, 3] / 2. / 250.,
                 data[:, 1] / 2.)


print((np.array(vf1), np.mean(vf1)))
print((np.array(vf15) / 1.5, np.mean(vf15) / 1.5))
plt.plot([0., 0.007], [0., 0.007 * 1800], 'k--', lw=1)
plt.plot([0., 0.007], [0., 0.007 * 1800 * 1.5], 'k--', lw=1)

plt.show()
