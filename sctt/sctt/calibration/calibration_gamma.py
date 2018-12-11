'''
Created on 24.11.2014

@author: Li Yingxiong
'''

from enthought.traits.api import HasTraits, Array, Instance, List, Float, Int, \
    Property, cached_property
from scipy.special import gamma as gamma_func
from scipy.interpolate import interp1d
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from quaducom.micro.resp_func.CB_clamped_rand_xi import CBClampedRandXi
from spirrid.spirrid import SPIRRID
from spirrid.rv import RV
from math import pi
from .tau_strength_dependence import interp_tau_shape, interp_tau_scale


plt.rc('text', usetex=True)
plt.rc('font', family='serif')

w_arr = np.linspace(0., 1., 50)
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

# for m in [7, 8, 9]:

tau_shape = []
residual = []
# tau_scale = []

# for m in [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]:
m = 7.0
for sV0 in [0.0080, 0.0085, 0.0090, 0.0095]:
    for scale in np.linspace(0.5, 1.5, 10):

        def func1(w_arr, k):

            tau = RV('gamma', shape=k, scale=scale, loc=0.)
            n_int = 500
            p_arr = np.linspace(0.5 / n_int, 1 - 0.5 / n_int, n_int)
            tau_arr = tau.ppf(p_arr) + 1e-10

            #     sV0 = sV0
            #     m = m
            r = 3.5e-3
            E_f = 180e3
            lm = 1000.

            def cdf(e, depsf, r, lm, m, sV0):
                '''weibull_fibers_cdf_mc'''
                s = ((depsf * (m + 1.) * sV0 ** m) /
                     (2. * pi * r ** 2.)) ** (1. / (m + 1.))
                a0 = (e + 1e-15) / depsf
                expfree = (e / s) ** (m + 1.)
                expfixed = a0 / \
                    (lm / 2.0) * (e / s) ** (m + 1.) * \
                    (1. - (1. - lm / 2.0 / a0) ** (m + 1.))
                mask = a0 < lm / 2.0
                exp = expfree * mask + \
                    np.nan_to_num(expfixed * (mask == False))
                return 1. - np.exp(- exp)

            T = 2. * tau_arr / r + 1e-10

            ef0cb = np.sqrt(w_arr[:, np.newaxis] * T[np.newaxis, :] / E_f)
            ef0lin = w_arr[:, np.newaxis] / lm + \
                T[np.newaxis, :] * lm / 4. / E_f
            depsf = T / E_f
            a0 = ef0cb / depsf
            mask = a0 < lm / 2.0
            e = ef0cb * mask + ef0lin * (mask == False)
            Gxi = cdf(e, depsf, r, lm, m, sV0)
            #             plt.plot(w_arr, np.average(Gxi, axis=1), label='s='+str(s)+'m='+str(m))
            mu_int = e * (1. - Gxi)
            sigma = mu_int * E_f

            return np.sum(sigma, axis=1) / n_int * (11. * 0.445) / 1000

        popt, pcov = curve_fit(func1, w_arr, sig_w)
        tau_shape.append(popt[0])
        residual.append(pcov[0][0])

        sigma = func1(w_arr, popt[0])
        plt.plot(
            w_arr, sigma, label='$m_\mathrm{f}=7, s_\mathrm{f}=$' + str(sV0) + ', $s_{tau}=$' + str(round(scale, 2)))

print(residual)

plt.plot(w_arr, sig_w, 'k--', lw=2, label='experiment')
plt.legend(loc='best', ncol=3)
plt.show()

#     lm = 1000.
#     spirrid.eps_vars = dict(w=w)
#     m = 9
#     spirrid.theta_vars = dict(tau=tau, E_f=Ef, V_f=V_f, r=r, m=m, sV0=sV0, lm=lm)
#     spirrid.n_int = n_int
#     sigma_c = spirrid.mu_q_arr / r ** 2
#     return sigma_c

#         popt, pcov = curve_fit(func1, w_arr, sig_w)
#
#         tau_shape.append(popt[0])
#         tau_scale.append(popt[1])
#
#         def func2(w_arr, k, theta):
#
#             tau = RV('gamma', shape=k, scale=theta, loc=0.)
#             n_int = 50
#             p_arr = np.linspace(0.5/n_int, 1 - 0.5/n_int, n_int)
#             tau_arr = tau.ppf(p_arr) + 1e-10
#
#             sV0 = s
#             m = mvalue
#             r = 3.5e-3
#             E_f = 180e3
#             lm =1000.
#
#             def cdf(e, depsf, r, lm, m, sV0):
#                 '''weibull_fibers_cdf_mc'''
#                 s = ((depsf*(m+1.)*sV0**m)/(2.*pi*r**2.))**(1./(m+1.))
#                 a0 = (e+1e-15)/depsf
#                 expfree = (e/s) ** (m + 1)
#                 expfixed = a0 / (lm/2.0) * (e/s) ** (m + 1) * (1.-(1.-lm/2.0/a0)**(m+1.))
#                 mask = a0 < lm/2.0
#                 exp = expfree * mask + np.nan_to_num(expfixed * (mask == False))
#                 return 1. - np.exp(- exp)
#
#
#             T = 2. * tau_arr / r + 1e-10
# k = np.sqrt(T/E_f)
# ef0cb = k*np.sqrt(w_arr)
#
#             ef0cb = np.sqrt(w_arr[:, np.newaxis] * T[np.newaxis, :]  / E_f)
#             ef0lin = w_arr[:, np.newaxis]/lm + T[np.newaxis, :]*lm/4./E_f
#             depsf = T/E_f
#             a0 = ef0cb/depsf
#             mask = a0 < lm/2.0
#             e = ef0cb * mask + ef0lin * (mask == False)
#             Gxi = cdf(e, depsf, r, lm, m, sV0)
# plt.plot(w_arr, np.average(Gxi, axis=1), label='s='+str(s)+'m='+str(m))
#             mu_int = e * (1.-Gxi)
#             sigma = mu_int*E_f
#             return sigma * (11. * 0.445) / 1000
#
# sigma = func1(w_arr, popt[0], popt[1])
#         sigma = func2(w_arr, popt[0], popt[1])
#
#         plt.clf()
#         plt.plot(w_arr, sigma, label='$s_{V_0}=$'+str(s)+', $m=$'+str(mvalue))
#         plt.xlim((0, 20))
# plt.legend()
#         savepath = 'D:\\m='+str(mvalue)+' sV0='+str(float(s))[:7]+'.png'
#         plt.savefig(savepath)


# print tau_shape
# print tau_scale
# plt.plot(w_arr, sig_w, '--', lw=2, label='single crack bridge test')
# plt.legend(loc='best', ncol=3)
# plt.ylim((0, 3.5))
# plt.xlabel('crack opening [mm]')
# plt.ylabel('fiber force [KN]')
# plt.show()

# class CalibrationGamma(HasTraits):
#
#     K = Array
#     Ef = Float(auto_set=False, enter_set=True, params=True)
#     lm = Float(auto_set=False, enter_set=True, params=True)
#     V_f = Float(.01, params=True)
#     r = Float(3.5e-3, params=True)
#     CS = Float(8.0, auto_set=False, enter_set=True, params=True)
#     sigmamu = Float(3.0, auto_set=False, enter_set=True, params=True)
#     w_hat = Float
#     test_xdata = Array
#     test_ydata = Array
#
#     interpolate_experiment = Property(depends_on='test_xdata, test_ydata')
#     @cached_property
#     def _get_interpolate_experiment(self):
#         return interp1d(self.test_xdata, self.test_ydata,
#                         bounds_error=False, fill_value=0.0)
#
#
#
#     def theta1(self):
#         mu_tau = 1.3 * self.r *self.sigmamu* (1.-self.V_f) / (2. * self.V_f * self.CS)
#         theta1 = mu_tau/self.K
#         return theta1
#
#     def theta2(self):
#         sigmaf_hat = self.interpolate_experiment(self.w_hat)
#         mu_sqrt_tau = sigmaf_hat / np.sqrt(2. * self.Ef * self.w_hat / self.r)
#         gamma_k = gamma_func(self.K+0.5)/gamma_func(self.K)
#         theta2 = (mu_sqrt_tau/gamma_k)**2
#         return theta2
#
# if __name__=='__main__':
#
#
#     cg = CalibrationGamma(K = np.linspace(0.05, 0.15, 20),
#                           Ef = 180e3,
#                           V_f = 0.01,
#                           w_hat = 0.025,
#                           CS = 10.,
#                           sigmamu = 3.4,
#                           r = 3.5e-3)


#     plt.plot(cg.K, cg.theta1())
#     plt.plot(cg.K, cg.theta2())
#     plt.show()
