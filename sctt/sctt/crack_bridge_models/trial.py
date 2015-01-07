'''
Created on 22.08.2014

@author: Li Yingxiong
'''
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import cumtrapz
from scipy.optimize import brute
from traits.api import HasTraits, \
    HasStrictTraits, Array
from scipy.optimize import brute, fmin_slsqp
from scipy.optimize import curve_fit
from calibration import Calibration
import os.path
from scipy.interpolate import interp1d
from scipy.stats import gamma as gam
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement import \
    ContinuousFibers
from stats.pdistrib.weibull_fibers_composite_distr import \
    WeibullFibers, fibers_MC
from spirrid.rv import RV
from quaducom.micro.resp_func.CB_clamped_rand_xi import CBClampedRandXi
from scipy.special import gammainc, gamma
from math import pi


tau_arr=np.logspace(np.log10(1e-5), 0.5, 500)
# tau_arr=np.linspace(1e-5, np.sqrt(10.), 500)

 
# tau_arr=np.linspace(1e-5, 0.5, 400)
 
w_arr = np.linspace(0.0, np.sqrt(3.), 401) ** 2
exp_data = np.zeros_like(w_arr)
home_dir = 'D:\\Eclipse\\'
for i in np.array([1, 2, 3, 4, 5]):
    path = [home_dir, 'git',  # the path of the data file
            'rostar',
            'scratch',
            'diss_figs',
            'CB'+str(i)+'.txt']
    filepath = os.path.join(*path)
#     exp_data = np.zeros_like(w_arr)
    file1 = open(filepath, 'r')
    cb = np.loadtxt(file1, delimiter=';')
    test_xdata = -cb[:, 2] / 4. - cb[:, 3] / 4. - cb[:, 4] / 2.
    test_ydata = cb[:, 1] / (11. * 0.445) * 1000
    interp = interp1d(test_xdata, test_ydata, bounds_error=False, fill_value=0.)
    exp_data += 0.2*interp(w_arr)
    
def get_sigma_tau_w(w_arr, tau, r, m, sV0, E_f):
    T = 2. * tau / r
    # scale parameter with respect to a reference volume
    s = ((T * (m + 1) * sV0 ** m) / (2. * E_f * pi * r ** 2)) ** (1. / (m + 1))
    ef0 = np.sqrt(w_arr * T  / E_f)
    Gxi = 1 - np.exp(-(ef0 / s) ** (m + 1))
    mu_int = ef0 * (1 - Gxi)
    sigma = mu_int *E_f
    return sigma

cali = Calibration(experi_data=exp_data,
                   w_arr=w_arr,
                   tau_arr=tau_arr,
                   m = 5.,
                   sV0=0.0085,
                   alpha = 0.,
                   shape = 0.176,
                   loc = 0.0057,
                   scale = 0.76,
                   bc=5,
                   sig_mu=3.4)

    
damage = cali.get_damage_portion(cali.sV0, 8.5e-2)
 
print np.sum(2*cali.tau_arr/cali.r*cali.tau_weights)
# # 
# print cali.sig_mu*(1-cali.V_f)/(cali.bc*cali.V_f)

# print np.sum(cali.tau_weights*cali.matrix_stress(9.5, 4.4e-2))


# idx = cali.tau_arr>=1
# print np.sum(cali.tau_weights[idx])
 
sV0 = cali.sV0
sigma = cali.get_sigma_tau_w(sV0)
sigma_avg = cali.sigma_c

# cali2 = Calibration(experi_data=10(sigma_avg-exp_data),
#                    w_arr=w_arr,
#                    tau_arr=tau_arr,
#                    m = 9.,
#                    sV0=0.005,
#                    alpha = 0.,
#                    shape = 0.176,
#                    loc = 0.0057,
#                    scale = 0.76,
#                    bc=1.85,
#                    sig_mu=3.4)



    
plt.subplot(221)
plt.plot(cali.w_arr, sigma_avg, '--', linewidth=2, label='calibrated response')
plt.plot(cali.w_arr, exp_data, label='average experimental data')

# exp_data1 = np.zeros_like(cali.w_arr)
# for i in np.array([1, 2, 3, 4, 5]):
#     path = [home_dir, 'git',  # the path of the data file
#             'rostar',
#             'scratch',
#             'diss_figs',
#             'CB'+str(i)+'.txt']
#     filepath = os.path.join(*path)
# #     exp_data = np.zeros_like(w_arr)
#     file1 = open(filepath, 'r')
#     cb = np.loadtxt(file1, delimiter=';')
#     test_xdata = -cb[:, 2] / 4. - cb[:, 3] / 4. - cb[:, 4] / 2. - cb[:, 1]/4000*1000/25e3*200
#     test_ydata = cb[:, 1] / (11. * 0.445) * 1000
#     interp = interp1d(test_xdata, test_ydata, bounds_error=False, fill_value=0.)
#     exp_data1 += 0.2*interp(cali.w_arr)
# print exp_data1
# plt.plot(cali.w_arr, exp_data1, '--', lw=4)
# cb = CBClampedRandXi()
# response_new = np.zeros_like(cali.w_arr)
# for i, tau in enumerate(cali.tau_arr):
#     for j, w in enumerate(cali.w_arr):
#         response_new[j] += cali.tau_weights[i]*cb(w, tau, 200e3, 0.01, 0.0035, 9.0, 0.005, 100.)
# print response_new
# plt.plot(cali.w_arr, response_new)

        
plt.legend(loc='best')
plt.xlabel('crack opening [mm]')
plt.ylabel('fiber stress [Mpa]')
#     plt.text(0.5, 0.5, 'm='+str(m)+', sV0='+str(float(sV0))[:7])
plt.subplot(222)
plt.bar(np.log10(cali.tau_arr), cali.tau_weights, 0.05)
# plt.plot(cali.tau_arr, cali.tau_weights)
# plt.xscale('log')
plt.xlabel('bond strength [Mpa]')
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)

# plt.figure()
# plt.subplot(121)
# plt.plot(cali.w_arr, sigma_avg, '--', linewidth=2, label='calibrated response')
# plt.plot(cali.w_arr, exp_data+cali2.sigma_c, label='average experimental data')
# 
# plt.legend(loc='best')
# plt.xlabel('crack opening [mm]')
# plt.ylabel('fiber stress [Mpa]')
# #     plt.text(0.5, 0.5, 'm='+str(m)+', sV0='+str(float(sV0))[:7])
# plt.subplot(122)
# plt.bar(cali.tau_arr, cali2.tau_weights, width=cali.tau_arr*0.05)
# # plt.plot(cali.tau_arr, cali.tau_weights)
# plt.xscale('log')
# plt.xlabel('bond strength [Mpa]')
# plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)






#     plt.plot(cali.tau_arr, x)
plt.subplot(223)
# plt.figure()
plt.plot(cali.w_arr, sigma)
plt.subplot(224)
x = cali.tau_weights
print np.sum(x)
for i, sigmai in enumerate(sigma.T):
    plt.plot(cali.w_arr, sigmai, color='0', lw='1.5', alpha=x[i] / np.max(x))
      
# plt.figure()
# rv=gam(0.176, loc=0.0057, scale=0.76)
# # tau1 = np.hstack((0, cali.tau_arr[0:-1]))
# # diff = (cali.tau_arr-tau1)/2
# # tau2 = diff+tau1
# #  
# #  
# gamma_weight = rv.cdf(cali.w_arr) - rv.cdf(np.hstack((0, cali.w_arr[0:-1])))
# # difference = cali.tau_weights-gamma_weight
# plt.bar(np.log10(cali.tau_arr), gamma_weight, width=0.02)
# plt.bar(np.log10(cali.tau_arr), cali.tau_weights, width=0.02)
# plt.figure()
# # plt.bar(np.log10(cali.tau_arr), difference, width=0.02)
# plt.plot(np.log10(tau_arr), rv.pdf(tau_arr))
  
  
# plt.figure()
# xi_shape = 9.0
# xi_scale = 0.005
# tau_scale = 0.8
# tau_shape = 0.20
# tau_loc = 0.005
# reinf1 = ContinuousFibers(r=3.5e-3,
#                           tau=RV('gamma', loc=tau_loc, scale=tau_scale, shape=tau_shape),
#                           V_f=0.01,
#                           E_f=200e3,
#                           xi=fibers_MC(m=xi_shape, sV0=xi_scale),
#                           label='carbon',
#                           n_int=500)
# tau = reinf1.results[4]
# plt.bar(np.log10(tau), np.ones_like(tau)*reinf1.stat_weights, width=0.02)
# plt.xlim((-3, 1))
# plt.ylim((0.0, 0.25))
# savepath = 'D:\cracking history\\1\\1.png'
# plt.savefig(savepath)
 
 
# plt.figure()
# cali_cdf = np.cumsum(cali.tau_weights)
# rv=gam(cali.shape, loc=cali.loc, scale=cali.scale)
# gamma_cdf = rv.cdf(cali.tau_arr)
# plt.plot(cali.tau_arr, cali_cdf, label = 'Calibrated distribution')
# plt.plot(cali.tau_arr, gamma_cdf, '--', label = 'Gamma distribution')
# plt.legend()
 
# savepath = 'D:\cracking history\\1\\2.png'
# plt.savefig(savepath)
 
 
plt.show()

