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
from etsproxy.util.home_directory import get_home_directory
import os.path

class Calibration(HasTraits):

#===============================================================================
# Parameters
#===============================================================================        
    # fiber stress data from experiment
    data = Array
    # the tau samples
    tau_arr = Array(input=True)
    # fiber radius
    r = Float(0.0035, auto_set=False, enter_set=True, input=True,
              distr=['uniform', 'norm'], desc='fiber radius')
    # fiber modulus
    E_f = Float(200e3, auto_set=False, enter_set=True, input=True,
                  distr=['uniform'])
    # shape parameter of the breaking strain distribution
    m = Float(9., auto_set=False, enter_set=True, input=True,
                  distr=['uniform'])
    # scale parameter of the breaking strain distribution
    sV0 = Float(0.0042, auto_set=False, enter_set=True, input=True,
                  distr=['uniform'])
    # fiber volume fraction
    V_f = Float(0.01, auto_set=False, enter_set=True, input=True,
              distr=['uniform'])
    # crack width
    w_arr = Array

#===============================================================================
# the crack width-fiber stress response
#===============================================================================        
    responses = Property(depends_on='r, E_f, m, sV0, V_f')
    @cached_property
    def _get_responses(self):
        T = 2. * self.tau_arr / self.r
        # scale parameter with respect to a reference volume
        s = ((T * (self.m + 1) * self.sV0 ** self.m) / (2. * self.E_f * pi * self.r ** 2)) ** (1. / (self.m + 1))
        ef0 = np.sqrt(self.w_arr[:, np.newaxis] * T[np.newaxis, :] / self.E_f)
        Gxi = 1 - np.exp(-(ef0 / s) ** (self.m + 1))
        mu_int = ef0 * (1 - Gxi)
        I = s * gamma(1 + 1. / (self.m + 1)) * gammainc(1 + 1. / (self.m + 1), (ef0 / s) ** (self.m + 1))
        mu_broken = I / (self.m + 1)
        sigma = (mu_int + mu_broken) * self.E_f
        return sigma

    def sumed_response(self, weights_arr):
        sigma = self.responses
        return np.sum(weights_arr * sigma, axis=1)

#===============================================================================
# optimization
#===============================================================================        
    def optimize_sV0(self):
        def residual(self, x):
            self.sV0 = float(x)
            sigma = self.responses
            sigma[0] = 1e6*np.ones_like(cali.tau_arr)
            self.data[0] = 1e6
            residual = nnls(sigma, self.data)[1]
            self.data[0] = 0.
            return residual
        sV0 = brute(residual, ((0.0001, 0.01),), Ns=10)
        print '1', sV0
        return sV0
    
    def tau_weights(self):
        sigma = self.responses
        sigma[0] = 1e6*np.ones_like(cali.tau_arr)
        data = self.data
        data[0] = 1e6
        x, y = nnls(sigma, data)
        sigma[0] = np.zeros_like(cali.tau_arr)
        data[0] = 0.
        return x

        

if __name__ == '__main__':

    w_arr = np.linspace(0.0, np.sqrt(8.), 401) ** 2
    
#===============================================================================
# read experimental data
#===============================================================================        
    home_dir = get_home_directory()
    path = [home_dir, 'git', # the path of the data file
            'rostar',
            'scratch',
            'diss_figs',
            'CB1.txt']
    filepath = os.path.join(*path)    
    data = np.zeros_like(w_arr)
    file1 = open(filepath, 'r')
    cb = np.loadtxt(file1, delimiter=';')
    test_xdata = -cb[:,2]/4. - cb[:,3]/4. - cb[:,4]/2.
    test_ydata = cb[:,1] / (11. * 0.445) * 1000
    interp = interp1d(test_xdata, test_ydata, bounds_error=False, fill_value=0.)
    data = interp(w_arr)
            
                
#     for m in np.linspace(2., 30., 29):
    
    cali = Calibration(data=data,
                       w_arr=w_arr,
                       tau_arr=np.logspace(np.log10(1e-5), np.log10(1), 200))
    
#===============================================================================
# optimize sV0 (minimize the difference using brute)
#===============================================================================        
    def residual(x):
        cali.sV0 = float(x)
        sigma = cali.responses
        sigma[0] = 1e6*np.ones_like(cali.tau_arr)
        data[0] = 1e6
        residual = nnls(sigma, data)[1]
        return residual
    sV0 = brute(residual, ((0.0001, 0.01),), Ns=10)
    cali.sV0 = float(sV0)

#===============================================================================
# output
#===============================================================================        

    sigma = cali.responses
    x = cali.tau_weights()
    sigma_avg = cali.sumed_response(x)
    
    plt.clf()
    plt.subplot(221)
    plt.plot(cali.w_arr, sigma_avg, '--', linewidth=2)
    plt.plot(cali.w_arr, data)
#     plt.text(0.5, 0.5, 'm='+str(m)+', sV0='+str(float(sV0))[:7])
    plt.subplot(222)
    plt.bar(np.log10(cali.tau_arr), x, width=0.02)
#     plt.plot(cali.tau_arr, x)
    plt.subplot(223)
    plt.plot(cali.w_arr, sigma)
    plt.subplot(224)
    for i, sigmai in enumerate(sigma.T):
        plt.plot(cali.w_arr, sigmai, color='0', lw='1.5', alpha=x[i]/np.max(x))
    plt.show()
    
#         savepath = 'F:\\parametric study\\cb_avg\\m='+str(m)+' sV0='+str(float(sV0))[:7]+'.png'
#         plt.savefig(savepath)





