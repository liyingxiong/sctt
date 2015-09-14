'''
Created on 10.10.2014

@author: Li Yingxiong
'''
from crack_bridge_models.random_bond_cb import RandomBondCB
import numpy as np
from scipy.interpolate import interp1d
import os.path
from reinforcements.fiber_bundle import FiberBundle
from stats.pdistrib.weibull_fibers_composite_distr import \
    WeibullFibers, fibers_MC
import matplotlib.pyplot as plt
from scipy.optimize import fmin, fmin_powell, fmin_slsqp
from calibration import Calibration


tau_arr=np.logspace(np.log10(1e-5), 0.5, 100)

w_arr = np.linspace(0.0, np.sqrt(8.), 401) ** 2
home_dir = 'D:\\Eclipse\\'
path = [home_dir, 'git',  # the path of the data file
        'rostar',
        'scratch',
        'diss_figs',
        'CB1.txt']
filepath = os.path.join(*path)
exp_data = np.zeros_like(w_arr)
file1 = open(filepath, 'r')
cb = np.loadtxt(file1, delimiter=';')
test_xdata = -cb[:, 2] / 4. - cb[:, 3] / 4. - cb[:, 4] / 2.
test_ydata = cb[:, 1] / (11. * 0.445) * 1000
interp = interp1d(test_xdata, test_ydata, bounds_error=False, fill_value=0.)
exp_data = interp(w_arr)

cali = Calibration(experi_data=exp_data,
                   w_arr=w_arr,
                   tau_arr=tau_arr,
                   m = 8,
                   sV0=0.0045)


def calibration_cb(tau_weights):
    
    sigma = cali.get_sigma_tau_w(0.0045)
    sig_avg = np.sum(tau_weights * sigma, axis=1)
    
    
    reinf = FiberBundle(r=0.0035,
              tau=tau_arr,
              tau_weights=tau_weights,
              V_f=0.01,
              E_f=200e3,
              xi=fibers_MC(m=8, sV0=0.0045))
 
    ccb = RandomBondCB(E_m=25e3,
                   reinforcement_lst=[reinf],
                   Ll=6.85,
                   Lr=6.85,
                   L_max = 100)
    ccb.max_sig_c(ccb.Ll, ccb.Lr)
    ccb.damage
    print ccb.E_m*ccb._epsm_arr[-1]
    residual = np.sum(np.abs(exp_data-sig_avg)) + \
                100000*np.abs(ccb.E_m*ccb._epsm_arr[-1]-3.5)
    print residual
    return residual


x = fmin(calibration_cb, cali.tau_weights, maxiter=3000)
 
# print x


sigma = cali.get_sigma_tau_w(0.0045)
sig_avg = np.sum(x * sigma, axis=1)

plt.figure()
plt.plot(w_arr, exp_data, '--')
plt.plot(w_arr, sig_avg)

reinf = FiberBundle(r=0.0035,
          tau=tau_arr,
          tau_weights=x,
          V_f=0.01,
          E_f=200e3,
          xi=fibers_MC(m=8, sV0=0.0045))

ccb1 = RandomBondCB(E_m=25e3,
                   reinforcement_lst=[reinf],
                   Ll=6.85,
                   Lr=6.85,
                   L_max = 100)
print ccb1.max_sig_c(ccb1.Ll, ccb1.Lr)
plt.figure()
plt.plot(ccb1._x_arr, ccb1.E_m*ccb1._epsm_arr)

plt.figure()
plt.plot(np.zeros_like(ccb1._epsf0_arr), ccb1._epsf0_arr, 'ro', label='maximum')
for i, depsf in enumerate(ccb1.sorted_depsf):
    epsf_x = np.maximum(ccb1._epsf0_arr[i] - depsf * np.abs(ccb1._x_arr), ccb1._epsm_arr)
    print np.trapz(epsf_x - ccb1._epsm_arr, ccb1._x_arr)
    if i == 0:
        plt.plot(ccb1._x_arr, epsf_x, color='blue', label='fibers')
    else:
        plt.plot(ccb1._x_arr, epsf_x, color='black')
plt.plot(ccb1._x_arr, ccb1._epsm_arr, lw=2, color='blue', label='matrix')
plt.legend(loc='best')
plt.ylabel('matrix and fiber strain [-]')
plt.ylabel('long. position [mm]')

plt.figure()
plt.subplot(211)
plt.bar(np.log10(cali.tau_arr), cali.tau_weights , width=0.02)
plt.subplot(222)
plt.bar(np.log10(cali.tau_arr), x, width=0.02)

plt.show()





    