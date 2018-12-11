'''
Created on Apr 30, 2015

@author: Yingxiong
'''
from crack_bridge_models.random_bond_cb import RandomBondCB
# from calibration import Calibration
import numpy as np
from scipy.interpolate import interp1d
import os.path
from reinforcements.fiber_bundle import FiberBundle
from stats.pdistrib.weibull_fibers_composite_distr import \
    WeibullFibers, fibers_MC
from composite_tensile_test import CompositeTensileTest
import matplotlib.pyplot as plt
from stats.misc.random_field.random_field_1D import RandomField
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement \
    import ContinuousFibers
from spirrid.rv import RV
from calibration.tau_strength_dependence import interp_tau_shape, interp_tau_scale
from calibration.matrix_strength_dependence import interp_m_shape
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.special import gamma
from scipy.optimize import bisect


def scale(shape):
    lp = 1.
    lc = 1000.
    sig_min = 2.72
    f = (lp / (lp + lc)) ** (1. / shape)
    return sig_min / (f * gamma(1. + 1. / shape))


def cal(k):

    home_dir = 'D:\\Eclipse\\'
    path1 = [home_dir, 'git',  # the path of the data file
             'rostar',
             'scratch',
             'diss_figs',
             'TT-4C-01.txt']
    filepath1 = os.path.join(*path1)
    data = np.loadtxt(filepath1, delimiter=';')
    eps_max = np.amax(-data[:, 2] / 2. / 250. - data[:, 3] / 2. / 250.)
    eps_arr = np.linspace(0, eps_max, 100)
    interp_exp = interp1d(-data[:, 2] / 2. / 250. - data[:, 3] / 2. / 250.,
                          data[:, 1] / 2., bounds_error=False, fill_value=0.)

    sig_exp = interp_exp(eps_arr)

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
    ctt = CompositeTensileTest(n_x=600,
                               L=300.,
                               cb=cb)

    n = 10
    s_m_arr = np.linspace(2.8, 3.8, n)

    lack_of_fit = np.zeros(10)
    crack_spacing = np.zeros(10)

    for i in range(10):
        m_m = bisect(lambda m: scale(m) - s_m_arr[i], 1., 1000.)

        print(m_m)

        random_field = RandomField(seed=False,
                                   lacor=1.,
                                   length=300.,
                                   nx=600,
                                   nsim=1,
                                   loc=.0,
                                   scale=s_m_arr[i],
                                   shape=m_m,
                                   distr_type='Weibull')
        ctt.sig_mu_x = random_field.random_field

        sig_c_i, z_x_i, BC_x_i, sig_c_u, n_cracks = ctt.get_cracking_history()
        load_arr = np.linspace(0, sig_c_u, 100)
        eps_c_arr = ctt.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, load_arr)
        interp_sim = interp1d(
            eps_c_arr, load_arr, bounds_error=False, fill_value=0.)
        sig_sim = interp_sim(eps_arr)
        lack_of_fit[i] += np.sum((sig_sim - sig_exp) ** 2) / k
        crack_spacing[i] += ctt.L / n_cracks / k

    return lack_of_fit, crack_spacing

if __name__ == '__main__':

    from joblib import Parallel, delayed
    import multiprocessing

    inputs = list(range(5))
    num_cores = multiprocessing.cpu_count()


#     n = 10
#     s_m_arr = np.linspace(2.8, 3.8, n)

#     lack_of_fit, crack_spacing = cal(5)
    results = Parallel(n_jobs=num_cores)(delayed(cal)(5) for i in inputs)

    print(results)


#     fig, ax1 = plt.subplots()
#     ax1.plot(s_m_arr, lack_of_fit, 'k--', label='lack of fit:SCM-TT')
#     ax1.ylabel('lack of fit')
#     plt.legend(loc='best')
#
#     ax2 = ax1.twinx()
#     ax2.plot(s_m_arr, crack_spacing, 'k', label='crack spacing')
#     ax2.xlabel('s_m')
#     ax2.ylabel('crack spacing')
#     plt.legend(loc='best')
#     plt.show()
