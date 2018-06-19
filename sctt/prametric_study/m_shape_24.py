'''
Created on 16.04.2016

@author: Yingxiong
'''
import os.path

from scipy.interpolate import interp1d

from composite_tensile_test import CompositeTensileTest
from crack_bridge_models.random_bond_cb import RandomBondCB
import numpy as np
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement \
    import ContinuousFibers
from spirrid.rv import RV
from stats.misc.random_field.random_field_1D import RandomField
from stats.pdistrib.weibull_fibers_composite_distr import \
    fibers_MC


reinf = ContinuousFibers(r=3.5e-3,
                         tau=RV(
                             'gamma', loc=0.001260, scale=1.440, shape=0.0539),
                         V_f=0.01,
                         E_f=180e3,
                         xi=fibers_MC(m=6.7, sV0=0.0076),
                         label='carbon',
                         n_int=500)
cb = RandomBondCB(E_m=25e3,
                  reinforcement_lst=[reinf],
                  n_BC=10,
                  L_max=300)
ctt = CompositeTensileTest(n_x=1000,
                           L=500.,
                           cb=cb)

s_m_arr = np.linspace(2.4, 3.8, 40)

fpath = 'D:\\data\\Tensile_test_multiple_cracking\\ss_curve_mm_24_1\\'

for s_m in s_m_arr:

    print s_m

    random_field = RandomField(seed=False,
                               lacor=1.,
                               length=500.,
                               nx=1000,
                               nsim=1,
                               loc=.0,
                               scale=s_m,
                               shape=24.,
                               distr_type='Weibull')
    ctt.sig_mu_x = random_field.random_field

    sig_c_i, z_x_i, BC_x_i, sig_c_u, n_cracks = ctt.get_cracking_history()
    load_arr = np.linspace(0, sig_c_u, 100)
    eps_c_arr = ctt.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, load_arr)

    np.savetxt(fpath + str(s_m) + '.txt', np.vstack((load_arr, eps_c_arr)))

#     plt.plot(load_arr, eps_c_arr)
#     plt.show()
