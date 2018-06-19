from quaducom.micro.resp_func.CB_clamped_rand_xi import CBClampedRandXi
from traits.api import HasTraits, Array, List, Float, Property, \
    cached_property
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import gammainc, gamma
from math import pi
from scipy.optimize import basinhopping, fmin, fmin_l_bfgs_b, fmin_cobyla, brute, nnls
from scipy.interpolate import interp1d
from numpy.linalg import solve, lstsq
# from etsproxy.util.home_directory import get_home_directory
import os.path
# from crack_bridge_models.random_bond_cb import RandomBondCB
from reinforcements.fiber_bundle import FiberBundle
from stats.pdistrib.weibull_fibers_composite_distr import \
    WeibullFibers, fibers_MC
from scipy.stats import gamma as gam

class CaliGamma(HasTraits):

#===============================================================================
# Parameters
#===============================================================================
    # fiber stress data from experiment
    experi_data = Array(float, value=[])
    # the tau samples
    tau_arr = Array(input=True)
    # fiber radius
    r = Float(0.0035, auto_set=False, enter_set=True, input=True,
              distr=['uniform', 'norm'], desc='fiber radius')
    # fiber modulus
    E_f = Float(200e3, auto_set=False, enter_set=True, input=True,
                  distr=['uniform'])
    # shape parameter of the breaking strain distribution
    m = Float(7., auto_set=False, enter_set=True, input=True,
                  distr=['uniform'])
    # fiber volume fraction
    V_f = Float(0.01, auto_set=False, enter_set=True, input=True,
              distr=['uniform'])
    # crack width descritization
    w_arr = Array
#     # boundary condition of saturated state
#     bc = Float
#     # matrix strength
#     sig_mu = Float
    # coefficient
    alpha = Float
    
    shape = Float
    scale = Float
    loc = Float
    sV0 = Float
    
    #===============================================================================
    # the crack width-fiber stress response
    #===============================================================================
    def get_sigma_tau_w(self, sV0):
        T = 2. * self.tau_arr / self.r
        # scale parameter with respect to a reference volume
        s = ((T * (self.m + 1) * sV0 ** self.m) / (2. * self.E_f * pi * self.r ** 2)) ** (1. / (self.m + 1))
        ef0 = np.sqrt(self.w_arr[:, np.newaxis] * T[np.newaxis, :] / self.E_f)
        Gxi = 1 - np.exp(-(ef0 / s) ** (self.m + 1))
        mu_int = ef0 * (1 - Gxi)
        I = s * gamma(1 + 1. / (self.m + 1)) * gammainc(1 + 1. / (self.m + 1), (ef0 / s) ** (self.m + 1))
#         mu_broken = I / (self.m + 1)
        mu_broken = 0.
        sigma = (mu_int + mu_broken) * self.E_f
        return sigma

    
    def optimize(self):
        
        a = brute(self.get_lack_of_fit, ((0.005, 0.007), (0.15, 0.25), (0.7, 0.9)))
        
        return a

if __name__ == '__main__':
    
    tau_arr=np.logspace(np.log10(1e-5), 0.5, 500)
     
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
     
    cali = CaliGamma(experi_data=exp_data,
                       w_arr=w_arr,
                       tau_arr=tau_arr,
                       m = 9,
                       sV0=0.005)
    
    def get_lack_of_fit(x):
        
        sigma = cali.get_sigma_tau_w(cali.sV0)

        rv=gam(x[0], loc=x[1], scale=x[2])
        gamma_weight = rv.cdf(cali.tau_arr) - rv.cdf(np.hstack((0, cali.tau_arr[0:-1])))

        sigma_c = np.sum(gamma_weight * sigma, axis=1)
        
        return np.sum((sigma_c-cali.experi_data)**2)
    
    a = brute(get_lack_of_fit, ((0.15, 0.25), (0.005, 0.007), (0.7, 0.9)), Ns=10, finish=None)
    
    print a
        
        
        

    