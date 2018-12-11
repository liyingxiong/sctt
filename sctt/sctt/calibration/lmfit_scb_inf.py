'''
Created on Apr 22, 2015

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

# for w_max in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.8,
# 2.2, 2.6, 3.0]:
# for m1 in [6., 7., 8., 9., 10.]:
w_max = 1.0
w_arr = np.linspace(0., 1.0, 1000)
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
#     plt.plot(w_arr, interp(w_arr), 'k', alpha=0.5)
    sig_w += 0.2 * interp(w_arr)


def response(shape, scale, loc, x):

    sV0 = 0.0076
    m = 6.7

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

plt.plot(w_arr, sig_w, '--')
sig_model = response(54e-3, 1.44, 12.6e-4, w_arr)
plt.plot(w_arr, sig_model)
plt.show()


def fcn2min(params, x, data):
    shape = params['shape'].value
    scale = params['scale'].value
    loc = params['loc'].value
#         m = params['f_shape'].value
#     sV0 = params['f_scale'].value
    sV0 = 0.0076
    m = 6.7

    tau = RV('gamma', shape=shape, scale=scale, loc=loc)
    n_int = 500
    p_arr = np.linspace(0.5 / n_int, 1 - 0.5 / n_int, n_int)
    tau_arr = tau.ppf(p_arr) + 1e-10
    r = 3.5e-3
    E_f = 182e3
    T = 2. * tau_arr / r
    # scale parameter with respect to a reference volume
    s = ((T * (m + 1.) * sV0 ** m) /
         (2. * E_f * pi * r ** 2)) ** (1. / (m + 1.))
    ef0 = np.sqrt(x[:, np.newaxis] * T[np.newaxis, :] / E_f)
    Gxi = 1 - np.exp(-(ef0 / s) ** (m + 1.))
    mu_int = ef0 * (1 - Gxi)
    sigma = mu_int * E_f

    return np.sum(sigma, axis=1) / n_int * (11. * 0.445) / 1000 - data

# min loc guarantee a < 500
r = 3.5e-3
E_f = 180e3
min_loc = E_f * w_max * r / (2 * 500 ** 2)
#     print ''
#     print ''
#     print 'w_max', w_max
#     print 'min_l', min_loc

# create a set of Parameters
params = Parameters()
params.add('scale', value=1., min=0.)
params.add('shape',   value=1.,  min=0.)
params.add('loc', value=0., min=min_loc)
#     params.add('f_shape', value=25., min=0)
# params.add('f_scale', value=0.0142, min=0)

# kcs = fcn2min(params, w_arr, sig_w)
# print np.sum(kcs ** 2)

# do fit
# available methods
result = minimize(
    fcn2min, params, method='lbfgsb', args=(w_arr, sig_w))

# print sV0
# print 'w_max', w_max
print(('lack of fit', result.chisqr))
print(('normalized', result.chisqr / np.sum(sig_w ** 2)))
print(params)

# calculate final result
#     final = sig_w + result.residual
#
#     w6 = np.linspace(0, 6, 200)
#     sig6 = fcn2min(params, w6, np.zeros_like(w6))
#
#     plt.cla()
#     plt.plot(
#         w_arr, final, label='model')
#     plt.plot(w_arr, sig_w, 'k--', lw=2, label='experimental')
#     plt.plot(w6, sig6)
#     plt.legend(loc='best')
#     path = 'D:\\fig\\' + str(m1) + '.png'
#     plt.savefig(path)
# plt.title('m_f=' + str(m))
# plt.figure()
# plt.plot(w_arr1, sig_w1, 'k--', lw=2, label='experimental')
# sig1 = fcn2min(params, w_arr1, sig_w1) + sig_w1
# plt.plot(w_arr1, sig1, 'k', lw=2, label='model')
# plt.show()
