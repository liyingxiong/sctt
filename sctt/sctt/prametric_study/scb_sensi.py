'''
Created on May 28, 2015

@author: Yingxiong
'''
import numpy as np
import os
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from spirrid.rv import RV
from math import pi
import time as t
from scipy.special import gamma

w_arr = np.linspace(0., 1.0, 100)
sig_w = np.zeros_like(w_arr)
home_dir = 'D:\\Eclipse\\'
for i in np.array([1, 2, 3, 4, 5]):
    path = [home_dir, 'git',  # the path of the data file
            'rostar',
            'scratch',
            'diss_figs',
            'CB' + str(i) + '.txt']
    filepath = os.path.join(*path)
#     exp_data = np.zeros_like(w_arr)
    file1 = open(filepath, 'r')
    cb = np.loadtxt(file1, delimiter=';')
    test_xdata = -cb[:, 2] / 4. - cb[:, 3] / 4. - cb[:, 4] / 2.
    test_ydata = cb[:, 1]
    interp = interp1d(
        test_xdata, test_ydata, bounds_error=False, fill_value=0.)
    sig_w += 0.2 * interp(w_arr)

plt.plot(w_arr, sig_w, 'k--', label='Experiment')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# for factor in [0.6, 0.8, 1.0, 1.2, 1.4]:


def fcn2min(x):
    """ model decaying sine wave, subtract data"""
    shape = 0.0539
    scale = 1.44
    loc = 0.00126
    m = 6.7
    sV0 = 0.0076

    tau = RV('gamma', shape=shape, scale=scale, loc=loc)
    n_int = 500
    p_arr = np.linspace(0.5 / n_int, 1 - 0.5 / n_int, n_int)
    tau_arr = tau.ppf(p_arr) + 1e-10
    r = 3.5e-3
    E_f = 180e3
    T = 2. * tau_arr / r
    # scale parameter with respect to a reference volume
    s = ((T * (m + 1.) * sV0 ** m) /
         (2. * E_f * pi * r ** 2)) ** (1. / (m + 1.))
    ef0 = np.sqrt(x[:, np.newaxis] * T[np.newaxis, :] / E_f)
    Gxi = 1 - np.exp(-(ef0 / s) ** (m + 1.))
    mu_int = ef0 * (1 - Gxi)
    sigma = mu_int * E_f

    return np.sum(sigma, axis=1) / n_int * (11. * 0.445) / 1000

plt.plot(w_arr, fcn2min(w_arr), label='Simulation')
# label='$l_\\tau$=' + str(factor) + '$l_\\tau^\star$')
#          label='$s_\mathrm{f}$=' + str(factor) + '$s_\mathrm{f}^\star$')
plt.legend(loc='best')
plt.show()
