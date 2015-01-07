'''
Created on 10.11.2014

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
# tau_arr=np.linspace(1e-5, 0.5, 500)

 
# tau_arr=np.linspace(1e-5, 0.5, 400)
 
w_arr = np.linspace(0.0, np.sqrt(8.), 401) ** 2
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

cali = Calibration(experi_data=exp_data,
                   w_arr=w_arr,
                   tau_arr=tau_arr,
                   m = 9.,
                   sV0=0.005,
                   conti_weight = 0,
                   conti_con = np.ones_like(tau_arr),
                   bc=15.,
                   sig_mu=3.)

temp = np.hstack((0, cali.tau_weights, 0))

continue_con  = np.zeros_like(cali.tau_weights)
for i in range(len(cali.tau_weights)):
    continue_con[i] = (temp[i]+temp[i+2])/2

cali.conti_weight = 1e4

for j in range(100):
    cali.conti_con = continue_con
    
    plt.clf()
    plt.subplot(121)
    plt.plot(cali.w_arr, cali.sigma_c, '--', linewidth=2, label='calibrated response')
    plt.plot(cali.w_arr, exp_data, label='average experimental data')
    plt.xlim((0,1))
    plt.subplot(122)
    plt.bar(cali.tau_arr, cali.tau_weights, width=cali.tau_arr*0.05)
    plt.xscale('log')
    plt.xlabel('bond strength [Mpa]')
    savepath = 'D:\cracking history\\1\\'+str(j)+'.png'
    plt.savefig(savepath)
    
    temp = np.hstack((0, cali.tau_weights, 0))

    continue_con  = np.zeros_like(cali.tau_weights)
    for k in range(len(cali.tau_weights)):
        continue_con[k] = (temp[k]+temp[k+2])/2

    
    



    

    
