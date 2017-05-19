'''
Created on May 8, 2015

@author: Yingxiong
'''
from joblib import Parallel, delayed
import multiprocessing
from crack_bridge_models.random_bond_cb import RandomBondCB
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


def calculate(eps_arr, sig_lst, sig_avg, k):

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
                               cb=cb)

    n = 40
    s_m_arr = np.linspace(2.4, 3.8, n)
    m_m_arr = np.linspace(10, 100, n)
    X, Y = np.meshgrid(s_m_arr, m_m_arr)
    x = X.flatten()
    y = Y.flatten()

    #lack_of_fit = np.zeros((n, n))
    #crack_spacing = np.zeros((n, n))
    #
    # for i in range(n):
    #    for j in range(n):

    random_field = RandomField(seed=False,
                               lacor=1.,
                               length=500.,
                               nx=1000,
                               nsim=1,
                               loc=.0,
                               scale=x[k],
                               shape=y[k],
                               distr_type='Weibull')
    ctt.sig_mu_x = random_field.random_field

    sig_c_i, z_x_i, BC_x_i, sig_c_u, n_cracks = ctt.get_cracking_history()
    load_arr = np.linspace(0, sig_c_u, 100)
    eps_c_arr = ctt.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, load_arr)
    interp_sim = interp1d(
        eps_c_arr, load_arr, bounds_error=False, fill_value=0.)
    sig_sim = interp_sim(eps_arr)
    lof1 = np.sum((sig_sim - sig_lst[0]) ** 2)
    lof2 = np.sum((sig_sim - sig_lst[1]) ** 2)
    lof3 = np.sum((sig_sim - sig_lst[2]) ** 2)
    lof4 = np.sum((sig_sim - sig_lst[3]) ** 2)
    lof5 = np.sum((sig_sim - sig_lst[4]) ** 2)
    lof_avg = np.sum((sig_sim - sig_avg) ** 2)
    crack_spacing = ctt.L / n_cracks

    return lof1, lof2, lof3, lof4, lof5, lof_avg, crack_spacing

if __name__ == '__main__':

    inputs = range(1600)
    #num_cores = multiprocessing.cpu_count()
    num_cores = 160

    eps_max_lst = []
    for j in range(5):
        filepath1 = 'D:\\data\\TT-6C-0' + str(j + 1) + '.txt'
        data = np.loadtxt(filepath1, delimiter=';')
        eps_max_lst.append(
            np.amax(-data[:, 2] / 2. / 250. - data[:, 3] / 2. / 250.))
    eps_max = np.amin(eps_max_lst)

    print eps_max

#     eps_max = 0.00575

    eps_arr = np.linspace(0, eps_max, 100)

    sig_lst = []
    for j in range(5):
        #     for j in [1]:
        filepath1 = 'D:\\data\\TT-6C-0' + str(j + 1) + '.txt'
        data = np.loadtxt(filepath1, delimiter=';')
        eps = -data[:, 2] / 2. / 250. - data[:, 3] / 2. / 250.
        sig = data[:, 1] / 2.
        if j == 1:
            print np.amax(eps[0:1500])
            interp_exp = interp1d(eps[0:1500],
                                  sig[0:1500], bounds_error=False, fill_value=0.)
        else:
            interp_exp = interp1d(eps, sig, bounds_error=False, fill_value=0.)

        sig_lst.append(interp_exp(eps_arr))
        plt.plot(eps_arr, interp_exp(eps_arr))

    sig_avg = np.sum(sig_lst, axis=0) / 5.
#     plt.plot(eps_arr, sig_avg)
    plt.show()

#     print len(sig_avg)

    lack_of_fit1 = np.zeros(1600)
    lack_of_fit2 = np.zeros(1600)
    lack_of_fit3 = np.zeros(1600)
    lack_of_fit4 = np.zeros(1600)
    lack_of_fit5 = np.zeros(1600)
    lack_of_fit_avg = np.zeros(1600)
    crack_spacing = np.zeros(1600)

    for i in range(5):
        print i
    #import time as t
    #t1 = t.time()
        results = Parallel(n_jobs=num_cores)(
            delayed(calculate)(eps_arr, sig_lst, sig_avg, k) for k in inputs)
        # print t.time()-t1

        A = np.array(results).T
        l1 = A[0]
        l2 = A[1]
        l3 = A[2]
        l4 = A[3]
        l5 = A[4]
        lavg = A[5]
        c = A[6]

        lack_of_fit1 += l1 / 5.
        lack_of_fit2 += l2 / 5.
        lack_of_fit3 += l3 / 5.
        lack_of_fit4 += l4 / 5.
        lack_of_fit5 += l5 / 5.
        lack_of_fit_avg += lavg / 5.
        crack_spacing += c / 5.
#
#    print [lack_of_fit]
#    print [crack_spacing]

    file1 = 'W:\\lof1.txt'
    np.savetxt(file1, lack_of_fit1.flatten(),
               fmt='%.4f', delimiter=',', newline='\n')

    file2 = 'W:\\lof2.txt'
    np.savetxt(file2, lack_of_fit2.flatten(),
               fmt='%.4f', delimiter=',', newline='\n')

    file3 = 'W:\\lof3.txt'
    np.savetxt(file3, lack_of_fit3.flatten(),
               fmt='%.4f', delimiter=',', newline='\n')

    file4 = 'W:\\lof4.txt'
    np.savetxt(file4, lack_of_fit4.flatten(),
               fmt='%.4f', delimiter=',', newline='\n')

    file5 = 'W:\\lof5.txt'
    np.savetxt(file5, lack_of_fit5.flatten(),
               fmt='%.4f', delimiter=',', newline='\n')

    file6 = 'W:\\lof_avg.txt'
    np.savetxt(file6, lack_of_fit_avg.flatten(),
               fmt='%.4f', delimiter=',', newline='\n')

    file7 = 'W:\\cs.txt'
    np.savetxt(file7, crack_spacing.flatten(),
               fmt='%.4f', delimiter=',', newline='\n')

#     from time import gmtime, strftime
#     s = strftime("%Y-%m-%d %H:%M:%S", gmtime())
#     text_file = open("W:\\Output.txt", "w")
#     text_file.write(s)
#     text_file.close()

    print results
    #print [np.sum(results, axis=0) / 6]
