'''
Created on Apr 21, 2015

@author: Yingxiong
'''
from lmfit import minimize, Parameters, Parameter, report_fit
import numpy as np
import os
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from spirrid.rv import RV
from math import pi
import time as t
from scipy.special import gamma

# for w_max in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]:
# for s_f in [0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012]:
# experimental data to be fitted
w_arr = np.linspace(0., 0.8, 100)
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


def fcn2min(shape, scale, loc, x):
    #     shape = params['shape'].value
    #     scale = params['scale'].value
    #     loc = params['loc'].value
    #     m = params['f_shape'].value
    #         sV0 = params['f_scale'].value
    sV0 = 0.0069
    m = 7.1

    tau = RV('gamma', shape=shape, scale=scale, loc=loc)
    n_int = 500
    p_arr = np.linspace(0.5 / n_int, 1 - 0.5 / n_int, n_int)
    tau_arr = tau.ppf(p_arr) + 1e-10
    r = 3.5e-3
    E_f = 180e3
#     lm = 1000
#
#     def cdf(e, depsf, r, lm, m, sV0):
#         '''weibull_fibers_cdf_mc'''
#         s = ((depsf * (m + 1.) * sV0 ** m) /
#              (2. * pi * r ** 2.)) ** (1. / (m + 1.))
#         a0 = (e + 1e-15) / depsf
#         expfree = (e / s) ** (m + 1.)
#         expfixed = a0 / \
#             (lm / 2.0) * (e / s) ** (m + 1) * \
#             (1. - (1. - lm / 2.0 / a0) ** (m + 1.))
#         print expfree
#         print expfixed
#         mask = a0 < lm / 2.0
#         exp = expfree * mask + \
#             np.nan_to_num(expfixed * (mask == False))
#         return 1. - np.exp(- exp)
#
#     T = 2. * tau_arr / r + 1e-10
#     ef0cb = np.sqrt(x[:, np.newaxis] * T[np.newaxis, :] / E_f)
#     ef0lin = x[:, np.newaxis] / lm + \
#         T[np.newaxis, :] * lm / 4. / E_f
#     depsf = T / E_f
#     a0 = ef0cb / depsf
#     mask = a0 < lm / 2.0
#     e = ef0cb * mask + ef0lin * (mask == False)
#     Gxi = cdf(e, depsf, r, lm, m, sV0)
#     mu_int = e * (1. - Gxi)
#     sigma = mu_int * E_f

    T = 2. * tau_arr / r
    # scale parameter with respect to a reference volume
    s = ((T * (m + 1.) * sV0 ** m) /
         (2. * E_f * pi * r ** 2)) ** (1. / (m + 1.))
    ef0 = np.sqrt(x[:, np.newaxis] * T[np.newaxis, :] / E_f)
    Gxi = 1 - np.exp(-(ef0 / s) ** (m + 1.))
    mu_int = ef0 * (1 - Gxi)
    sigma = mu_int * E_f

    return np.sum(sigma, axis=1) / n_int * (11. * 0.445) / 1000

if __name__ == '__main__':

    plt.plot(w_arr, sig_w, 'k--', label='experiment')
    model = fcn2min(
        0.098859511867768868, 0.61630329230447289, 5.1058291625771801e-06, w_arr)
    plt.plot(w_arr, model, 'k', label='model')
    plt.legend(loc='best')
    plt.show()
