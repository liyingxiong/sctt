'''
Created on 09.10.2014

@author: Li Yingxiong
'''

from crack_bridge_models.random_bond_cb import RandomBondCB
from calibration import Calibration
import numpy as np
from scipy.interpolate import interp1d
import os.path
from reinforcements.fiber_bundle import FiberBundle
from stats.pdistrib.weibull_fibers_composite_distr import \
    WeibullFibers, fibers_MC
import matplotlib.pyplot as plt

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


cb_strength = []
m_arr = np.linspace(6, 12, 7)


cali = Calibration(experi_data=exp_data,
                   w_arr=w_arr,
                   tau_arr=np.logspace(np.log10(1e-5), 0.5, 200),
                   m = 8,
                   sV0=0.0045)

reinf = FiberBundle(r=0.0035,
              tau=cali.tau_arr,
              tau_weights = cali.tau_weights,
              V_f=0.01,
              E_f=200e3,
              xi=fibers_MC(m=cali.m, sV0=cali.sV0))

ccb = RandomBondCB(E_m=25e3,
                   reinforcement_lst=[reinf],
                   Ll=6.85,
                   Lr=6.85,
                   L_max = 100)

ccb.max_sig_c(ccb.Ll, ccb.Lr)

print ccb.w

# cb_strength.append(ccb.max_sig_c(6.85, 6.85)[0])

# sig = []
# for w in np.linspace(1e-15, 2, 200):
#     sig.append(ccb.sig_c(w))
# plt.plot(np.linspace(1e-15, 2, 200), sig)
# plt.show()
# ccb.damage
plt.plot(ccb._x_arr, ccb.E_m*ccb._epsm_arr)
plt.xlabel('m')
plt.ylabel('cb_strength(Mpa)')
plt.show()
    




