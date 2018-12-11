'''
Created on Jun 21, 2018

@author: liyin
'''
'''
Created on Jun 21, 2018

@author: liyin
'''
from sctt.crack_bridge_models.random_bond_cb import RandomBondCB
import numpy as np
from scipy.interpolate import interp1d
import os.path
from sctt.reinforcements.fiber_bundle import FiberBundle
from stats.pdistrib.weibull_fibers_composite_distr import \
    WeibullFibers, fibers_MC
from sctt.composite_tensile_test import CompositeTensileTest
import matplotlib.pyplot as plt
from stats.misc.random_field.random_field_1D import RandomField
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement \
    import ContinuousFibers
from spirrid.rv import RV
from sctt.calibration.matrix_strength_dependence import interp_m_shape


reinf = ContinuousFibers(r=3.5e-3,
                         tau=RV(
                             'gamma', loc=0.001260, scale=1.440, shape=0.0539),
                         V_f=0.015,
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
                           cb=cb,
                           save_cracking = True)
mean = 2.8 * 1.34
stdev = 0.01

random_field = RandomField(seed=False,
                           lacor=1.,
                           length=500.,
                           nx=1000,
                           nsim=1,
                           mean=mean,
                           stdev=stdev,
                           distr_type='Gauss')

ctt.sig_mu_x = random_field.random_field

sig_c_i, z_x_i, BC_x_i, sig_c_u, n_cracks = ctt.get_cracking_history()
load_arr = np.unique(np.hstack((np.linspace(0, sig_c_u, 100), sig_c_i)))
print(load_arr)
eps_c_arr = ctt.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, load_arr)
eps_c_i = np.interp(np.hstack((sig_c_i, sig_c_u)), load_arr, eps_c_arr)
cs = 500. / (np.arange(len(z_x_i)) + 1)
cs[cs > 120.] = 120.
cs = np.hstack((cs, cs[-1]))

print(BC_x_i)


plt.figure()
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)

ax1.plot(eps_c_arr, load_arr, lw=2)
ax1.plot(eps_c_i[1:-1], sig_c_i[1::], 'o')
ax2.plot(eps_c_i, cs, drawstyle='steps', lw=2)

plt.show()



