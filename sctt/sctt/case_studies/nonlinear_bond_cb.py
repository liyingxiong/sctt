'''
Created on 19.03.2017

@author: Yingxiong
'''
from multiple_cracking.cb import NonLinearCB
from stats.misc.random_field.random_field_1D import RandomField
import matplotlib.pyplot as plt
import numpy as np
from sctt.composite_tensile_test import CompositeTensileTest


if __name__ == '__main__':

    cb = NonLinearCB(
        slip=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
        bond=[0., 10., 20., 30., 40., 100.],
        n_BC=20)

    random_field = RandomField(seed=False,
                               lacor=1.,
                               length=500,
                               nx=1000,
                               nsim=1,
                               loc=.0,
                               shape=60.,
                               scale=2.5,
                               distr_type='Weibull')

    ctt = CompositeTensileTest(n_x=1000,
                               L=500,
                               cb=cb,
                               sig_mu_x=random_field.random_field)

    sig_c_i, z_x_i, BC_x_i, sig_c_u, n_crack = ctt.get_cracking_history()
    load_arr = np.unique(np.hstack((np.linspace(0, sig_c_u, 100), sig_c_i)))
    eps_c_arr = ctt.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, load_arr)
    plt.plot(eps_c_arr, load_arr, 'k', lw=2, label='v_f=1.5%')
    plt.show()
