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
# from scipy.interpolate import interp1d

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
    E_f = Float(180e3, auto_set=False, enter_set=True, input=True,
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
    bc = Float
#     # matrix strength
    sig_mu = Float
    # coefficient
    alpha = Float
    
    shape = Float
    scale = Float
    loc = Float
    
    conti_con = Array
    conti_weight = Float
    
    
    #===============================================================================
    # the crack width-fiber stress response
    #===============================================================================
    def get_sigma_tau_w(self, sV0):
        T = 2. * self.tau_arr / self.r
        # scale parameter with respect to a reference volume
        s = ((T * (self.m + 1) * sV0 ** self.m) / (2. * self.E_f * pi * self.r ** 2)) ** (1. / (self.m + 1))
        ef0 = np.sqrt(self.w_arr[:, np.newaxis] * T[np.newaxis, :]  / self.E_f)
        Gxi = 1 - np.exp(-(ef0 / s) ** (self.m + 1))
        mu_int = ef0 * (1 - Gxi)
        I = s * gamma(1 + 1. / (self.m + 1)) * gammainc(1 + 1. / (self.m + 1), (ef0 / s) ** (self.m + 1))
#         mu_broken = I / (self.m + 1)
        mu_broken = 0.
        sigma = (mu_int + mu_broken) * self.E_f
        return sigma

    #===============================================================================
    # the crack width-fiber stress response
    #===============================================================================
    def get_damage_portion(self, sV0, w):
        T = 2. * self.tau_arr / self.r
        # scale parameter with respect to a reference volume
        s = ((T * (self.m + 1) * sV0 ** self.m) / (2. * self.E_f * pi * self.r ** 2)) ** (1. / (self.m + 1))
        ef0 = np.sqrt(self.w_arr[:, np.newaxis] * T[np.newaxis, :] / self.E_f)
        Gxi = 1 - np.exp(-(ef0 / s) ** (self.m + 1))
        i = np.sum(self.w_arr<=w)-1
        return Gxi[i]

    #===============================================================================
    # the debonding length
    #===============================================================================
    def get_debonding_length(self, w):
        return np.sqrt(0.5*w*self.E_f*self.r/self.tau_arr)

    #===============================================================================
    # the matrix stress at x from crack position
    #===============================================================================
    def matrix_stress(self, x, w):
        def H(x):
            return x>0
        T = 2*self.tau_arr/self.r
        sig_f0 = np.sqrt(w*self.E_f*T)
        a = self.get_debonding_length(w)
        damage = self.get_damage_portion(self.sV0, w)
        transfered_stress = (sig_f0*H(x-a)+x*T*H(a-x))*damage
        return transfered_stress*self.V_f*(1-self.V_f)
        
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
        
#     sV0 = Property(Float, depends_on='r, E_f, m, V_f')
#     '''Value of sV0 minimizing the lack of fit.
#     '''
#     @cached_property
#     def _get_sV0(self):
#         residual = lambda sv0: self.get_lack_of_fit(sv0)
#         sV0 = brute(residual, ((0.0001, 0.01),), Ns=10)
#         return float(sV0)
    sV0 = Float

    #===========================================================================
    # Postprocessing for calibrated tau distribution  and sv0
    #===========================================================================
    sigma_c = Property(depends_on='r, E_f, m, V_f, conti_con, conti_weight')
    @cached_property
    def _get_sigma_c(self):
        sigma = self.get_sigma_tau_w(self.sV0)
        return np.sum(self.tau_weights * sigma, axis=1)

    tau_weights = Property(Float, depends_on='r, E_f, m, V_f, conti_con, conti_weight')
    @cached_property
    def _get_tau_weights(self):
        sigma = (1-self.alpha)*self.get_sigma_tau_w(self.sV0)
        sigma[0] = 1e6 * np.ones_like(self.tau_arr)

#         sigma1 = (1-self.alpha)*self.get_sigma_tau_w(self.sV0)
#         sigma1[0] = 1e6 * np.ones_like(self.tau_arr)
        
#         idx = np.sum(self.experi_data<=400)
#         sigma = sigma1[0]
# 
#         
#         damage = self.get_damage_portion(self.sV0, 8.5e-2)
#         print damage
        T = 1e4*2. * self.tau_arr / self.r
        sigma = np.vstack((sigma, T)) #constraint for initial slope of matrix stress

#         for x in np.linspace(1, 7, 100):
#             matrix_stress = self.matrix_stress(x, 4.4e-2)
#             sigma = np.vstack((sigma, 1e3*matrix_stress))
        
        #gamma constraint
        rv=gam(0.11757297717, loc=0., scale=0.651310376269)
        gamma_weight = rv.cdf(self.tau_arr) - rv.cdf(np.hstack((0, self.tau_arr[0:-1])))
        n_factor = np.amax(self.experi_data)/np.amax(gamma_weight)
        diagonal = self.alpha*n_factor*np.eye(len(self.tau_arr))
        sigma = np.vstack((sigma, diagonal)) #constraint for scatter of tau_weights

#         diagonal = self.conti_weight*np.eye(len(self.tau_arr))
#         sigma = np.vstack((sigma, diagonal))
        
#         tau_sqr = 1e3*self.tau_arr**2
#         sigma = np.vstack((sigma, tau_sqr))
        
        data = (1-self.alpha)*np.copy(self.experi_data)
        data[0] = 1e6
#         data = np.array([1e6])

        data = np.hstack((data, 1e4*self.sig_mu*(1-self.V_f)/(self.bc*self.V_f)))

#         data = np.hstack((data, 1e3*((self.sig_mu*(1-self.V_f)/(self.bc*self.V_f)*0.5*self.r)**2+0.2)))
#         mean_tau = 5e2*np.ones_like(self.tau_arr)*self.sig_mu*(1-self.V_f)/(self.bc*self.V_f)*self.r*0.5

#         rv=gam(0.22, loc=0.007, scale=0.8)
#         gamma_weight = rv.cdf(self.tau_arr) - rv.cdf(np.hstack((0, self.tau_arr[0:-1])))
        
#         
#         stress = np.sqrt(np.linspace(1, 7, 100))/np.sqrt(7)*3.5 
#         data = np.hstack((data, 1e3*stress))
        
        #gamma constraint
        data = np.hstack((data, self.alpha*n_factor*gamma_weight))
        
#         data = np.hstack((data, self.conti_weight*self.conti_con))

        
        x, y = nnls(sigma, data)
        
        return x

if __name__ == '__main__':

    w_arr = np.linspace(0.0, np.sqrt(3.), 401) ** 2

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
    m=4
    sv0_lst = []
    sig_saturated_max1 = []
    w_lst = []
    m_lst = []
    y_lst = []
    sig_bc = []
     
    while m<=30:
 
        m += 1
        print m
        cali = Calibration(experi_data=exp_data,
                           w_arr=w_arr,
                           tau_arr=np.logspace(np.log10(1e-4), 0.5, 400),
                           bc=6.85,
                           sig_mu=3.4,
                           sV0 = 0.0045,
                           m=m
                           )
    
    #===============================================================================
    # output
    #===============================================================================
    
        sV0 = cali.sV0
        
        sigma = cali.get_sigma_tau_w(sV0)
        sigma_avg = cali.sigma_c
        
#         print 'stdv', np.dot(cali.tau_weights, (cali.tau_arr)**2)-(cali.sig_mu*(1-cali.V_f)/(cali.bc*cali.V_f)*0.5*cali.r)**2
    
        plt.clf()
        plt.subplot(321)
        plt.plot(cali.w_arr, sigma_avg, '--', linewidth=2)
        plt.plot(cali.w_arr, exp_data)
        plt.xlim(0, 8)
        plt.ylim(0, 700)
    #     plt.text(0.5, 0.5, 'm=' + str(m) + ', sV0=' + str(float(sV0))[:7])
        plt.subplot(322)
        plt.bar(np.log10(cali.tau_arr), cali.tau_weights , width=0.02)
        tau_e = cali.sig_mu*(1-cali.V_f)/(cali.bc*cali.V_f)*0.5*cali.r
        plt.plot((np.log10(tau_e), np.log10(tau_e)), (0, 1.), '--')
        plt.xlim(-5, 1)
        plt.ylim(0, 1)
    #     plt.plot(cali.tau_arr, x)
        plt.subplot(323)
        plt.plot(cali.w_arr, sigma)
        plt.xlim(0, 8)
        plt.ylim(0, 3000)
        print 'mean T:'+str(cali.sig_mu*(1-cali.V_f)/(cali.bc*cali.V_f))
        print 'weighted sum of T:'+str(np.sum(cali.tau_weights*2*cali.tau_arr/cali.r))
    
    
        plt.subplot(324)
        x = cali.tau_weights
        for i, sigmai in enumerate(sigma.T):
            plt.plot(cali.w_arr, sigmai, color='0', lw='1.5', alpha=x[i] / np.max(x))
    #     plt.show()
        plt.xlim(0, 8)
        plt.ylim(0, 3000)

        
#         tau_ind = np.nonzero(x)
#          
#         reinf1 = FiberBundle(r=0.0035,
#                             tau=cali.tau_arr[tau_ind],
#                             tau_weights = x[tau_ind],
#                             V_f=0.01,
#                             E_f=200e3,
#                             xi=fibers_MC(m=cali.m, sV0=float(sV0))
#                             )
#         cb1 =  RandomBondCB(E_m=25e3,
#                            reinforcement_lst=[reinf1],
#                            Ll=6.85,
#                            Lr=6.85)
#  
#  
#         reinf2 = FiberBundle(r=0.0035,
#                             tau=cali.tau_arr[tau_ind],
#                             tau_weights = x[tau_ind],
#                             V_f=0.015,
#                             E_f=200e3,
#                             xi=fibers_MC(m=cali.m, sV0=float(sV0))
#                             )
#         cb2 = RandomBondCB(E_m=25e3,
#                            reinforcement_lst=[reinf2],
#                            Ll=4.03,
#                            Lr=4.03)
#          
#         sig_saturated1 = np.zeros_like(cali.w_arr)
#         sig_saturated2 = np.zeros_like(cali.w_arr)
#         for i, w in enumerate(cali.w_arr):
#             sig_saturated1[i] = cb1.sig_c(w)
#             sig_saturated2[i] = cb2.sig_c(w)
#          
#         plt.subplot(325)
#         ind = np.nonzero(sig_saturated2)
#         plt.plot(cali.w_arr[ind], sig_saturated1[ind]/0.01, label='0.01')
#         plt.plot(cali.w_arr[ind], sig_saturated2[ind]/0.015, label='0.015')
#         plt.legend()
#         plt.xlim(0, 0.3)
#         plt.ylim(0, 3000)
         
#         plt.show()
         
 
#         savepath = 'D:\\parametric study\\cb_1\\m='+str(m)+' sV0='+str(float(sV0))[:7]+'.png'
        sv0_lst.append(sV0)
#         max_ind = np.argmax(sig_saturated)
#         sig_saturated_max.append(sig_saturated[max_ind])
#         w_lst.append(cali.w_arr[max_ind])
        m_lst.append(m)
        y_lst.append(cali.get_lack_of_fit(sV0))
         
         
         
         
#         cb1.Ll = 250
#         cb1.Lr = 250
#          
#         cb2.Ll = 250
#         cb2.Lr = 250
#          
#         sig_interp1 = np.zeros_like(cali.w_arr)
#         sig_interp2 = np.zeros_like(cali.w_arr)
#          
#         for i, w in enumerate(cali.w_arr):
#             sig_interp1[i] = cb1.sig_c(w)
#             sig_interp2[i] = cb2.sig_c(w)
#  
#  
#  
#          
#  
#          
#          
#          
#         plt.subplot(326)
#         ind1 = np.argmax(sig_interp1[1:])
#         f1 = interp1d(sig_interp1[1:ind1], cali.w_arr[1:ind1])
#          
#         ind2 = np.argmax(sig_interp2[1:])
#         f2 = interp1d(sig_interp2[1:ind2], cali.w_arr[1:ind2])
#          
#         cb1.w = float(f1(5))
#         print cb1.w
#         cb1.damage
         
#         cb2.w = float(f2(5))
#         print cb2.w
#         cb2.damage
#  
#         plt.plot(cb1._x_arr, cb1.E_m*cb1._epsm_arr, label='1@5')
#         plt.plot(cb2._x_arr, cb2.E_m*cb2._epsm_arr, label='1.5@5')
         
#         cb1.w = float(f1(10))
#         print cb1.w
#         cb1.damage
#          
#         cb2.w = float(f2(10))
#         print cb2.w
#         cb2.damage
#  
#         plt.plot(cb1._x_arr, cb1.E_m*cb1._epsm_arr, label='1@10')
#         plt.plot(cb2._x_arr, cb2.E_m*cb2._epsm_arr, label='15@10')
 
#         plt.legend()
#         plt.xlim(0, 200)
#         plt.ylim(0, 10.)
#         sig_bc.append(cb.E_m*cb._epsm_arr[-1])
         
        savepath = 'D:\\parametric study\\number of tau\\m='+str(m)+'.png'
        plt.savefig(savepath)
 
 
     
    plt.clf()
    plt.subplot(221)
    plt.plot(m_lst, sv0_lst)
    plt.subplot(222)
#     plt.plot(m_lst, sig_saturated_max)
    plt.subplot(223)
    plt.plot(m_lst, y_lst)
    plt.subplot(224)
    plt.plot(m_lst, w_lst)
    plt.show()






