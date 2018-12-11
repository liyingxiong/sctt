'''
Identify the first cracking stress according to the derivatives of the stress-strain diagram.
@author: Yingxiong
'''
import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from lmfit import minimize, Parameters, Parameter, report_fit


eps_max_lst = []
for j in range(5):
    filepath1 = 'D:\\data\\TT-6C-0' + str(j + 1) + '.txt'
    data = np.loadtxt(filepath1, delimiter=';')
    eps_max_lst.append(
        np.amax(-data[:, 2] / 2. / 250. - data[:, 3] / 2. / 250.))
eps_max = np.amin(eps_max_lst)

eps_arr = np.linspace(0, eps_max, 1000)

sig_lst = []

params = Parameters()
params.add('k1',   value=0.,  min=0.)
params.add('k2', value=0., min=0.)
params.add('k3', value=0., min=0.)
params.add('a', value=0., min=5e-5, max=0.001)
params.add('b', value=0., min=0.001, max=0.005)


def f(params, x, data):
    k1 = params['k1'].value
    k2 = params['k2'].value
    k3 = params['k3'].value
    a = params['a'].value
    b = params['b'].value
    return k1 * x * (x <= a) + (k1 * a + k2 * (x - a)) * (x > a) * (x <= b) + (k1 * a + k2 * (b - a) + k3 * (x - b)) * (x > b) - data

for j in range(5):
    filepath1 = 'D:\\data\\TT-6C-0' + str(j + 1) + '.txt'
    data = np.loadtxt(filepath1, delimiter=';')
    interp_exp = interp1d(-data[:, 2] / 2. / 250. - data[:, 3] / 2. / 250.,
                          data[:, 1] / 2., bounds_error=False, fill_value=0.)
#     interp_exp = UnivariateSpline(-data[:, 2] / 2. / 250. - data[:, 3] / 2. / 250.,
#                                      data[:, 1] / 2., k=3)
#     popt, pcov = curve_fit(f, eps_arr, interp_exp(eps_arr))
#     k1, k2, a = popt
#     e = f(eps_arr, k1, k2, a)
#     plt.plot(eps_arr, e)
    sig_lst.append(interp_exp(eps_arr))
    result = minimize(
        f, params, method='powell', args=(eps_arr, interp_exp(eps_arr)))

    final = interp_exp(eps_arr) + result.residual
    print(params)
#     print interp_exp(params['a'].value)
    print((interp_exp(params['a'].value) * 25 / (25 * 0.985 + 2.7)))
    if j == 1:
        plt.plot(eps_arr, interp_exp(eps_arr))
        plt.plot(eps_arr, final, 'k--')


#     plt.plot(eps_arr, interp_exp(eps_arr))

sig_avg = np.sum(sig_lst, axis=0) / 5.
#     plt.plot(eps_arr, sig_avg)


for k in range(5):

    dsig = np.gradient(sig_lst[k])
#     plt.plot(eps_arr, dsig)


plt.show()
