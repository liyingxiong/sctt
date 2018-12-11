'''
Created on Jan 22, 2015

@author: Li Yingxiong
'''
from .tau_strength_dependence import interp_tau_shape, interp_tau_scale
import numpy as np
from crack_bridge_models.random_bond_cb import RandomBondCB
from stats.pdistrib.weibull_fibers_composite_distr import \
    WeibullFibers, fibers_MC
from scipy.optimize import brentq
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement import \
    ContinuousFibers
from spirrid.rv import RV
from matplotlib import pyplot as plt

m_arr = [6., 7., 8., 9., 10., 11.]


sV0_arr = []
for m in m_arr:
    def strength(sV0):
        scale = float(interp_tau_scale(sV0, m))
        shape = float(interp_tau_shape(sV0, m))
#         print scale, shape
        reinf = ContinuousFibers(r=3.5e-3,
                                 tau=RV(
                                     'gamma', loc=0.0, scale=scale, shape=shape),
                                 V_f=0.015,
                                 E_f=180e3,
                                 xi=fibers_MC(m=m, sV0=sV0),
                                 label='carbon',
                                 n_int=500)

        ccb = RandomBondCB(E_m=25e3,
                           reinforcement_lst=[reinf],
                           Ll=7.,
                           Lr=350.,
                           L_max=400,
                           n_BC=20)
        return ccb.max_sig_c(1., 375.)[0]

    def strength0(sV0):
        return strength(sV0) - 20.59

    sV0_arr.append(brentq(strength0, 1e-15, 0.020))

print(sV0_arr)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(m_arr, sV0_arr)
plt.xlabel('$\psi_\mathrm{f}$')
plt.ylabel('$s_\mathrm{f}$')
plt.show()
