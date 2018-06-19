'''
Created on Mar 22, 2015

@author: Yingxiong
'''
from sctt_aramis import CTTAramis
import numpy as np
from matplotlib import pyplot as plt
from stats.pdistrib.weibull_fibers_composite_distr import \
    WeibullFibers, fibers_MC
from crack_bridge_models.random_bond_cb import RandomBondCB
import os
from scipy.interpolate import interp1d
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement \
    import ContinuousFibers
from spirrid.rv import RV
from scipy.stats import variation


# 4 layers
plt.figure()
for i in range(4):
    # for i in [3]:
    homedir = 'D:\\data\\Tensile_test_multiple_cracking\\'
    path = [homedir, 'test' + str(i + 1) + '.txt']
    filepath = os.path.join(*path)
    data = np.loadtxt(filepath, delimiter=',')
    stress = np.sort(data[1, :] / 2.)

    if i == 0:
        plt.plot(np.hstack((0., stress, 13.5)), np.hstack((0, np.arange(
            len(stress) + 1) / 120.)), color='0.5', drawstyle='steps', label='experiment')
    else:
        plt.plot(np.hstack((0., stress, 13.5)), np.hstack((0, np.arange(
            len(stress) + 1) / 120.)), color='0.5', drawstyle='steps')

stress_1 = np.array(
    [3.320,   4.111,   4.465, 5.499,   6.177,  7.948,   8.210, 10.733, 10.930])

plt.plot(np.hstack((0., stress_1, 13.5)), np.hstack((0, np.arange(
    len(stress_1) + 1) / 120.)), 'k', lw=2, drawstyle='steps', label='crack tracing algorithm')

# 6 layers
plt.figure()
for i in range(4):
    # for i in [3]:
    homedir = 'D:\\data\\Tensile_test_multiple_cracking\\'
    path = [homedir, 'test6' + str(i + 1) + '.txt']
    filepath = os.path.join(*path)
    data = np.loadtxt(filepath, delimiter=',')
    stress = np.sort(data[1, :] / 2.)

    if i == 0:
        plt.plot(np.hstack((0., stress, 21.00)), np.hstack((0, np.arange(
            len(stress) + 1) / 120.)), color='0.5', drawstyle='steps', label='experiment')
    else:
        plt.plot(np.hstack((0., stress, 21.00)), np.hstack((0, np.arange(
            len(stress) + 1) / 120.)), color='0.5', drawstyle='steps')

stress_1 = np.array([4.16120678,   4.8967324,   5.52525786,
                     6.17354454,   6.5216327,   7.69627393,   8.89836948,
                     10.00924641,  11.29460214,  12.82018033,  15.42674058,
                     16.17140557,  19.53467005])

plt.plot(np.hstack((0., stress_1, 21.00)), np.hstack((0, np.arange(
    len(stress_1) + 1) / 120.)), 'k', lw=2, drawstyle='steps', label='crack tracing algorithm')

plt.show()
