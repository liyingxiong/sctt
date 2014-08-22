from etsproxy.traits.api import \
    HasStrictTraits, Instance, Int, Float, List, Array, Property, \
    cached_property
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq, minimize_scalar, fmin, brute
from random_fields.simple_random_field import SimpleRandomField
from crack_bridge_models.constant_bond_cb import ConstantBondCB
from crack_bridge_models.stochastic_cb import StochasticCB
from crack_bridge_models.representative_cb import RepresentativeCB
from util.traits.either_type import EitherType
from etsproxy.traits.ui.api import \
    View, Item, Group
from stats.pdistrib.weibull_fibers_composite_distr import \
    WeibullFibers, fibers_MC
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.hom_CB_elastic_mtrx \
    import CompositeCrackBridge
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement \
    import ContinuousFibers
from spirrid.rv import RV
from stats.misc.random_field.random_field_1D import RandomField
from matplotlib import pyplot as plt
from crack_bridge_models.random_bond_cb import RandomBondCB, FiberBundle
from calibration import Calibration
import os.path


class CompositeTensileTest(HasStrictTraits):
    
    cb = EitherType(klasses=[ConstantBondCB, \
                             RepresentativeCB]) #crack bridge model

#=============================================================================
# Discretization of the specimen
#=============================================================================    
    n_x = Int(501) #number of material points
    L = Float(1000.) #the specimen length - mm
    x = Property(depends_on='n_x, L') #coordinates of the material points
    @cached_property
    def _get_x(self):
        return np.linspace(0, self.L, self.n_x)
    sig_mu_x = Array()

#=============================================================================
# Description of cracked specimen
#=============================================================================    
    y = List([]) #the list to record the crack positions
    z_x = Property(depends_on='x, y')
    '''the array containing the distances from each material point to its 
    nearest crack position
    '''
    @cached_property
    def _get_z_x(self):
        try:
            y = np.array(self.y)
            z_grid = np.abs(self.x[:, np.newaxis] - y[np.newaxis, :])
            return np.amin(z_grid, axis=1)
        except ValueError: #no cracks exist
            return np.ones_like(self.x)*2*self.L
        
    BC_x = Property(depends_on='x, y')
    '''the array containing the boundary condition for each material point
    '''
    @cached_property
    def _get_BC_x(self):
        try:
            y = np.sort(self.y)
            d = (y[1:] - y[:-1]) / 2.0
            #construct a piecewise function for interpolation
            xp = np.hstack([0, y[:-1]+d, self.x[-1]])
            L_left = np.hstack([y[0], d, np.NAN])
            L_right = np.hstack([d, self.L-y[-1], np.NAN])
            f = interp1d(xp, np.vstack([L_left, L_right]), kind='zero')
            return f(self.x)
        except IndexError:
            return np.vstack([np.zeros_like(self.x), np.zeros_like(self.x)])
        
#=============================================================================
# Determine the cracking load level
#=============================================================================

    def get_sig_c_z(self, sig_mu, z, Ll, Lr):
        '''Determine the composite remote stress initiating a crack 
        at position z'''
        fun = lambda sig_c: sig_mu - self.cb.get_sig_m_z(z, Ll, Lr, sig_c)
        try:
        # search the cracking stress level between zero and ultimate
        # composite stress
            return brentq(fun, 0, self.cb.sig_cu)
        
        except ValueError:
        # solution not found 
            try:
            # find the load level corresponding to the maximum matrix stress
            # (non-monotonic)
                sig_m = lambda sig_c: -self.cb.get_sig_m_z(z, Ll, Lr, sig_c)
                sig_max = brute(sig_m, ((0., self.cb.sig_cu),), Ns=6)[0]
                return brentq(fun, 0, sig_max)
            
            except ValueError:
            # shielded zone, impossible to crack, return the ultimate stress 
                return self.cb.sig_cu
         
    get_sig_c_x_i = np.vectorize(get_sig_c_z)
     
    def get_sig_c_i(self):
        '''Determine the new crack position and level of composite stress
        '''
        # for each material point find the load factor initiating a crack
        sig_c_x_i = self.get_sig_c_x_i(self, self.sig_mu_x, \
                                       self.z_x, self.BC_x[0], self.BC_x[1])
        # get the position with the minimum load factor
        y_idx = np.argmin(sig_c_x_i)
        y_i = self.x[y_idx]
        sig_c_i = sig_c_x_i[y_idx]
        return sig_c_i, y_i

#=============================================================================
# determine the crack history
#=============================================================================
    def get_cracking_history(self):
        '''Trace the response crack by crack.
        '''
        z_x_lst = [self.z_x]
        BC_x_lst = [self.BC_x]
        sig_c_lst = [0.0]
        while True:
            sig_c_i, y_i = self.get_sig_c_i()
            print sig_c_i, y_i
            self.y.append(y_i)
            sig_c_lst.append(sig_c_i)
            z_x_lst.append(np.array(self.z_x))
            BC_x_lst.append(np.array(self.BC_x))            
            if sig_c_i == self.cb.sig_cu: break
        print 'cracking history determined'
        self.y = []
        return np.array(sig_c_lst), np.array(z_x_lst), BC_x_lst

#=============================================================================
# post processing
#=============================================================================
    def get_eps_c_i(self, sig_c_i, z_x_i, BC_x_i):
        '''For each cracking level calculate the corresponding
        composite strain eps_c.
        '''
        return np.array([np.trapz(self.get_eps_f_x(sig_c, z_x, BC_x[0], \
                                                   BC_x[1]), self.x) / self.L
                         for sig_c, z_x, BC_x in zip(sig_c_i, z_x_i, BC_x_i)
                         ])
    
    def get_eps_f_x(self, sig_c, z_x, Ll_arr, Lr_arr):
        '''function to evaluate specimen reinforcement strain profile
        at given load level and crack distribution
        ''' 
        eps_f = np.zeros_like(self.x)
        z_x_map = np.argsort(z_x)
        eps_f[z_x_map] = self.cb.get_eps_f_z(z_x[z_x_map], Ll_arr, Lr_arr, sig_c)
        return eps_f
    
    def get_sig_m_x(self, sig_c, z_x, Ll_arr, Lr_arr):
        '''function to evaluate specimen matrix stress profile
        at given load level and crack distribution
        ''' 
        eps_m = np.zeros_like(self.x)
        z_x_map = np.argsort(z_x)
        eps_m[z_x_map] = self.cb.get_sig_m_z(z_x[z_x_map], Ll_arr[z_x_map], \
                                             Lr_arr[z_x_map], sig_c)
        return eps_m
    
    def get_eps_c_arr(self, sig_c_i, z_x_i, BC_x_i, load_arr):
        '''function to evaluate the average specimen strain array corresponding 
        to the given load_arr
        '''
        eps_arr = np.ones_like(load_arr)
        for i, load in enumerate(load_arr):
            idx = np.searchsorted(sig_c_i, load) - 1
            z_x = z_x_i[idx]
            if np.any(z_x == 2*self.L):
                eps_arr[i] = load / self.cb.E_c
            else:
                BC_x = BC_x_i[idx]
                eps_arr[i] = np.trapz(self.get_eps_f_x(load, z_x, BC_x[0], \
                                      BC_x[1]), self.x) / self.L
            
            #save the cracking history
            plt.clf()                          
            sig_m = self.get_sig_m_x(load, z_x, BC_x[0], BC_x[1])
            plt.plot(self.x, sig_m)
            plt.plot(self.x, self.sig_mu_x)
            savepath = 'D:\cracking history\\1\\load_step'+str(i+1)+'.png'
            plt.savefig(savepath)
            
        return eps_arr
    
    def get_w_dist(self, sig_c_i, z_x_i, BC_x_i):
        '''function for evaluate the crack with
        '''
        w_dist = []
        for sig_c, z_x, BC_x in zip(sig_c_i, z_x_i, BC_x_i):
            eps_f_x = self.get_eps_f_x(sig_c, z_x, BC_x[0], BC_x[1])
            eps_m_x = self.get_sig_m_x(sig_c, z_x, BC_x[0], BC_x[1]) / self.cb.E_m
            y = self.x[z_x == 0]
            if not y.size:
                continue
            distance = np.abs(self.x[:, np.newaxis] - y[np.newaxis, :])
            nearest_crack = y[np.argmin(distance, axis=1)]
            w_arr = np.array([np.trapz( (eps_f_x[nearest_crack == y_i] - \
                                         eps_m_x[nearest_crack == y_i]), \
                                       self.x[nearest_crack == y_i] ) \
                               for y_i in y])
            w_dist.append(w_arr)
        return w_dist
    
    def save_cracking_history(self, sig_c_i, z_x_lst, BC_x_lst):
        '''save the cracking history'''
        plt.clf()
        plt.subplot(411)
        i = len(z_x_lst)
        BC_x = BC_x_lst[i-2]                          
        sig_m = self.get_sig_m_x(sig_c_i, z_x_lst[i-2], BC_x[0], BC_x[1])
        print sig_c_i
        plt.plot(self.x, sig_m)
        plt.plot(self.x, self.sig_mu_x)
        
        plt.subplot(412)
        plt.plot(self.x, self.z_x)

        plt.subplot(413)
        plt.plot(self.x, self.BC_x[0])
        
        plt.subplot(414)
        plt.plot(self.x, self.BC_x[1])
        savepath = 'D:\cracking history\\1\\BC'+str(len(self.y))+'.png'
        plt.savefig(savepath)

               
            
    
    view = View(Item('L', show_label=False),
                Item('n_x', show_label=False),
                Item('cb', show_label=False),
                buttons=['OK', 'Cancel'])

#=============================================================================
# output
#=============================================================================    
if __name__ == '__main__':
    
    #=============================================================================
    # calibration
    #=============================================================================
    w_arr = np.linspace(0.0, np.sqrt(8.), 401) ** 2    
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
    cali = Calibration(experi_data=exp_data,
                       w_arr=w_arr,
                       tau_arr=np.logspace(np.log10(1e-5), np.log10(1), 200))
    tau_ind = np.nonzero(cali.tau_weights)
    
    
    
    reinf = FiberBundle(r=0.0035,
                  tau=cali.tau_arr[tau_ind],
                  tau_weights = cali.tau_weights[tau_ind],
                  V_f=0.1,
                  E_f=200e3,
                  xi=fibers_MC(m=cali.m, sV0=cali.sV0))


    
#     reinf = ContinuousFibers(r=0.0035,
#                           tau=RV('weibull_min', loc=0.0, shape=1., scale=1.),
#                           V_f=0.01,
#                           E_f=180e3,
#                           xi=fibers_MC(m=2.0, sV0=0.003),
#                           label='carbon',
#                           n_int=200)
     
#     cb1 = CompositeCrackBridge(E_m=25e3,
#                                reinforcement_lst=[reinf])
    cb =  RandomBondCB(E_m=25e3,
                       reinforcement_lst=[reinf])
    
    rcb = RepresentativeCB(ccb=cb)
         
#     cbcb = ConstantBondCB(n_z = 500,
#                           T = 30)
        
#     rf = SimpleRandomField(n_x = 501,
#                            mean = 4,
#                            deviation = 1.)
    
    random_field = RandomField(seed=False,
                           lacor=1.,
                           length=100,
                           nx=501,
                           nsim=1,
                           loc=.0,
                           shape=50.,
                           scale=3.4,
                           distr_type='Weibull')
        
    ctt = CompositeTensileTest(n_x = 501,
                               L = 100,
                               cb=rcb,
                               sig_mu_x= random_field.random_field)
    
    print np.amin(ctt.sig_mu_x)
    
    sig_c_i, z_x_i, BC_x_i = ctt.get_cracking_history()
    eps_c_i = ctt.get_eps_c_i(sig_c_i, z_x_i, BC_x_i)
    
#     np.savetxt('D:\cracking history\\1\\cracking_loads.txt', sig_c_i)
#     np.savetxt('D:\cracking history\\1\\dmin.txt', z_x_i)
#     np.savetxt('D:\cracking history\\1\\BC.txt', BC_x_i)


    
        
    load_arr = np.linspace(0, ctt.cb.sig_cu, 200)
    eps_c_arr = ctt.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, load_arr)
    w_dist = ctt.get_w_dist(sig_c_i, z_x_i, BC_x_i)
      
#     ctt.configure_traits()
      
    plt.subplot(2, 2, 1)
    plt.plot(eps_c_i, sig_c_i)
    plt.plot([0.0, ctt.cb.sig_cu/ctt.cb.E_c], [0.0, ctt.cb.sig_cu])
    plt.plot(eps_c_arr, load_arr)
#     plt.plot([0.0, ctt.cb.eps_fu], [0.0, ctt.cb.sig_cu])
      
    plt.subplot(2, 2, 2)
    for i in range(1, len(z_x_i)):
        plt.plot(ctt.x, ctt.get_eps_f_x(sig_c_i[i], z_x_i[i], BC_x_i[i][0], BC_x_i[i][1]))
        plt.plot(ctt.x, ctt.get_sig_m_x(sig_c_i[i], z_x_i[i], BC_x_i[i][0], BC_x_i[i][1]) / ctt.cb.E_m)
    plt.ylim(ymin=0.0)
    
    plt.subplot(2, 2, 3)
    plt.plot(ctt.x, z_x_i[-1])
    plt.plot(ctt.x, ctt.sig_mu_x)
    plt.plot(ctt.x, ctt.get_sig_m_x(sig_c_i[-2], z_x_i[-2], BC_x_i[-2][0], BC_x_i[-2][1]))
    plt.hist(w_dist[-2])
    
    plt.subplot(2, 2, 4)
    plt.plot(ctt.x, ctt.sig_mu_x)
    for i in range(1, len(z_x_i)):
        plt.plot(ctt.x, ctt.get_sig_m_x(sig_c_i[i], z_x_i[i], BC_x_i[i][0], BC_x_i[i][1]))
    plt.ylim(ymin=0.0)

#     plt.subplot(2, 2, 2)
#     BC_x = BC_x_i[-2]
#     plt.plot(ctt.x, ctt.get_sig_m_x(sig_c_i[-2], z_x_i[-2], BC_x[0], BC_x[1]))
#     plt.plot(ctt.x, rf.field)
   
    plt.show()

            
    