'''
Created on Mar 13, 2015

@author: Li Yingxiong
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


def fcn2min(params, x, data):
    """ model decaying sine wave, subtract data"""
    shape = params['shape'].value
    scale = params['scale'].value
#     loc = params['loc'].value
#     m = params['f_shape'].value
#         sV0 = params['f_scale'].value
    sV0 = 0.007
    m = 7.

    tau = RV('gamma', shape=shape, scale=scale, loc=0.)
    n_int = 500
    p_arr = np.linspace(0.5 / n_int, 1 - 0.5 / n_int, n_int)
    tau_arr = tau.ppf(p_arr) + 1e-10
    r = 3.5e-3
    E_f = 180e3
    lm = np.inf

    def cdf(e, depsf, r, lm, m, sV0):
        '''weibull_fibers_cdf_mc'''
        s = ((depsf * (m + 1.) * sV0 ** m) /
             (2. * pi * r ** 2.)) ** (1. / (m + 1.))
        a0 = (e + 1e-15) / depsf
        expfree = (e / s) ** (m + 1)
        expfixed = a0 / \
            (lm / 2.0) * (e / s) ** (m + 1) * \
            (1. - (1. - lm / 2.0 / a0) ** (m + 1.))
        mask = a0 < lm / 2.0
        exp = expfree * mask + \
            np.nan_to_num(expfixed * (mask == False))
        return 1. - np.exp(- exp)

    T = 2. * tau_arr / r + 1e-10

    ef0cb = np.sqrt(x[:, np.newaxis] * T[np.newaxis, :] / E_f)
    ef0lin = x[:, np.newaxis] / lm + \
        T[np.newaxis, :] * lm / 4. / E_f
    depsf = T / E_f
    a0 = ef0cb / depsf
    mask = a0 < lm / 2.0
    e = ef0cb * mask + ef0lin * (mask == False)
    Gxi = cdf(e, depsf, r, lm, m, sV0)
    mu_int = e * (1. - Gxi)
    sigma = mu_int * E_f

    return np.sum(sigma, axis=1) / n_int * (11. * 0.445) / 1000 - data

# create a set of Parameters
params = Parameters()
params.add('shape',   value=1.,  min=0.)
params.add('scale', value=1., min=0.)
# params.add('loc', value=0., min=0.)
# params.add('f_shape', value=25., min=0)
#     params.add('f_scale', value=0.0142, min=0)

# kcs = fcn2min(params, w_arr, sig_w)
# print np.sum(kcs ** 2)

# do fit
# available methods
result = minimize(
    fcn2min, params, method='nelder', args=(w_arr, sig_w))

# print sV0
# print 'w_max', w_max
print 'lack of fit', result.chisqr
print params

# calculate final result
final = sig_w + result.residual

plt.cla()
plt.plot(
    w_arr, final, label='model')
plt.plot(w_arr, sig_w, 'k--', lw=2, label='experimental')
plt.legend(loc='best')
# path = 'D:\\fig\\' + str(w_max) + '.png'
# plt.savefig(path)
# plt.title('m_f=' + str(m))
# plt.figure()
# plt.plot(w_arr1, sig_w1, 'k--', lw=2, label='experimental')
# sig1 = fcn2min(params, w_arr1, sig_w1) + sig_w1
# plt.plot(w_arr1, sig1, 'k', lw=2, label='model')
plt.show()
