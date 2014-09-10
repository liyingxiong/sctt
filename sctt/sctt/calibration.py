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
from crack_bridge_models.random_bond_cb import RandomBondCB
from reinforcements.fiber_bundle import FiberBundle
from stats.pdistrib.weibull_fibers_composite_distr import \
    WeibullFibers, fibers_MC


class Calibration(HasTraits):

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
    m = Float(9., auto_set=False, enter_set=True, input=True,
                  distr=['uniform'])
    # fiber volume fraction
    V_f = Float(0.01, auto_set=False, enter_set=True, input=True,
              distr=['uniform'])
    # crack width descritization
    w_arr = Array
    # boundary condition of saturated state
    bc = Float
    # matrix strength
    sig_mu = Float
    
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

    #===============================================================================
    # optimization
    #===============================================================================
    def get_lack_of_fit(self, sV0):
        '''Residual function evaluating the lack of fit between the experimental data
        and the weighted sum of the tau levels.
        The function returns the array of non-negative weights delivering the best fit
        and the value of the lack of fit.
        '''
        sigma = self.get_sigma_tau_w(sV0)
        sigma[0] = 1e6 * np.ones_like(self.tau_arr)
        T = 1e3*2. * self.tau_arr / self.r
        sigma = np.vstack((sigma, T))
        data = np.copy(self.experi_data)
        data[0] = 1e6
        data = np.hstack((data, 1e3*self.sig_mu*(1-self.V_f)/(self.bc*self.V_f)))
        x, y = nnls(sigma, data)        
        return y
        
    sV0 = Property(Float, depends_on='r, E_f, m, V_f')
    '''Value of sV0 minimizing the lack of fit.
    '''
    @cached_property
    def _get_sV0(self):
        residual = lambda sv0: self.get_lack_of_fit(sv0)
        sV0 = brute(residual, ((0.0001, 0.01),), Ns=10)
        return float(sV0)

    #===========================================================================
    # Postprocessing for calibrated tau distribution  and sv0
    #===========================================================================
    sigma_c = Property(depends_on='r, E_f, m, V_f')
    @cached_property
    def _get_sigma_c(self):
        sigma = self.get_sigma_tau_w(self.sV0)
        return np.sum(self.tau_weights * sigma, axis=1)

    tau_weights = Property(Float, depends_on='r, E_f, m, V_f')
    @cached_property
    def _get_tau_weights(self):
        sigma = self.get_sigma_tau_w(self.sV0)
        sigma[0] = 1e6 * np.ones_like(self.tau_arr)
        data = np.copy(self.experi_data)
        data[0] = 1e6
        x, y = nnls(sigma, data)
        return x

if __name__ == '__main__':

    w_arr = np.linspace(0.0, np.sqrt(8.), 401) ** 2

#===============================================================================
# read experimental data
#===============================================================================
#     home_dir = get_home_directory()
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


#     for m in np.linspace(2., 30., 29):
    n=0
    sv0_lst = []
    sig_saturated_max = []
    w_lst = []
    n_lst = []
    y_lst = []
    sig_bc = []
    
    while n<=500:

        n += 10
        print n
        cali = Calibration(experi_data=exp_data,
                           w_arr=w_arr,
                           tau_arr=np.logspace(np.log10(1e-5), np.log10(1), n),
                           bc=4.03,
                           sig_mu=3.4
                           )
    
    #===============================================================================
    # output
    #===============================================================================
    
        sV0 = cali.sV0
        sigma = cali.get_sigma_tau_w(sV0)
        sigma_avg = cali.sigma_c
    
        plt.clf()
        plt.subplot(321)
        plt.plot(cali.w_arr, sigma_avg, '--', linewidth=2)
        plt.plot(cali.w_arr, exp_data)
        plt.xlim(0, 8)
        plt.ylim(0, 700)
    #     plt.text(0.5, 0.5, 'm=' + str(m) + ', sV0=' + str(float(sV0))[:7])
        plt.subplot(322)
        plt.bar(np.log10(cali.tau_arr), cali.tau_weights , width=0.02)
        plt.xlim(-5, 1)
        plt.ylim(0, 1)
    #     plt.plot(cali.tau_arr, x)
        plt.subplot(323)
        plt.plot(cali.w_arr, sigma)
        plt.xlim(0, 8)
        plt.ylim(0, 3000)

        plt.subplot(324)
        x = cali.tau_weights
        for i, sigmai in enumerate(sigma.T):
            plt.plot(cali.w_arr, sigmai, color='0', lw='1.5', alpha=x[i] / np.max(x))
    #     plt.show()
        plt.xlim(0, 8)
        plt.ylim(0, 3000)

        
        tau_ind = np.nonzero(x)
        reinf = FiberBundle(r=0.0035,
                            tau=cali.tau_arr[tau_ind],
                            tau_weights = x[tau_ind],
                            V_f=0.015,
                            E_f=200e3,
                            xi=fibers_MC(m=cali.m, sV0=float(sV0))
                            )
        cb =  RandomBondCB(E_m=25e3,
                           reinforcement_lst=[reinf],
                           Ll=4.03,
                           Lr=4.03)
        sig_saturated = np.zeros_like(cali.w_arr)
        for i, w in enumerate(cali.w_arr):
            sig_saturated[i] = cb.sig_c(w)
        
        plt.subplot(325)
        ind = np.nonzero(sig_saturated)
        plt.plot(cali.w_arr[ind], sig_saturated[ind])
        plt.xlim(0, 0.3)
        plt.ylim(0, 15)
        
#         plt.show()
        

#         savepath = 'D:\\parametric study\\cb_1\\m='+str(m)+' sV0='+str(float(sV0))[:7]+'.png'
        sv0_lst.append(sV0)
        max_ind = np.argmax(sig_saturated)
        sig_saturated_max.append(sig_saturated[max_ind])
        w_lst.append(cali.w_arr[max_ind])
        n_lst.append(n)
        y_lst.append(cali.get_lack_of_fit(sV0))
        
        plt.subplot(326)
        cb.w = cali.w_arr[max_ind]
        cb.Ll = 1e5
        cb.Lr = 1e5
        cb.damage
        plt.plot(cb._x_arr, cb.E_m*cb._epsm_arr)
        plt.xlim(0, 200)
        plt.ylim(0, 6.)
        sig_bc.append(cb.E_m*cb._epsm_arr[-1])
        
        savepath = 'D:\\parametric study\\number of tau\\n='+str(n)+'.png'
        plt.savefig(savepath)


    
    plt.clf()
    plt.subplot(221)
    plt.plot(n_lst, sv0_lst)
    plt.subplot(222)
    plt.plot(n_lst, sig_saturated_max)
    plt.subplot(223)
    plt.plot(n_lst, y_lst)
    plt.subplot(224)
    plt.plot(n_lst, w_lst)
    plt.show()





