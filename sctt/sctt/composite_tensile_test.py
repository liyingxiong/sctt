from etsproxy.traits.api import \
    HasStrictTraits, Instance, Int, Float, List, Array, Property, \
    cached_property
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from random_fields.simple_random_field import SimpleRandomField
from crack_bridge_models.constant_bond_cb import ConstantBondCB
from crack_bridge_models.stochastic_cb import StochasticCB
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

class CompositeTensileTest(HasStrictTraits):
    
    cb = EitherType(klasses=[ConstantBondCB, \
                             StochasticCB]) #crack bridge model

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
        '''Determine the composite remote stress initiating a crack at position z
        '''
        fun = lambda sig_c: sig_mu - self.cb.get_sig_m_z(z, Ll, Lr, sig_c)
        try:
        # search the cracking stress level between zero and ultimate composite stress
            return brentq(fun, 0, 1000)
        except ValueError:
            # solution not found (shielded zone) return the ultimate composite stress
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
            self.y.append(y_i)
            sig_c_lst.append(sig_c_i)
            z_x_lst.append(np.array(self.z_x))
            BC_x_lst.append(np.array(self.BC_x))
            print sig_c_i, y_i
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
        eps_m[z_x_map] = self.cb.get_sig_m_z(z_x[z_x_map], Ll_arr, Lr_arr, sig_c)
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
        return eps_arr
    
    def get_w_dist(self, sig_c_i, z_x_i, BC_x_i):
        '''function for evaluate the crack with
        '''
        w_dist = []
        for sig_c, z_x, BC_x in zip(sig_c_i, z_x_i, BC_x_i):
#             try:
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
#             except:
#                 print y
        return w_dist
               
            
    
    view = View(Item('L', show_label=False),
                Item('n_x', show_label=False),
                Item('cb', show_label=False),
                buttons=['OK', 'Cancel'])

#=============================================================================
# output
#=============================================================================    
if __name__ == '__main__':
    reinf = ContinuousFibers(r=0.0035,
                          tau=RV('weibull_min', loc=0.0, shape=1., scale=1.),
                          V_f=0.01,
                          E_f=180e3,
                          xi=fibers_MC(m=2.0, sV0=0.003),
                          label='carbon',
                          n_int=200)
     
    cb1 = CompositeCrackBridge(E_m=25e3,
                               reinforcement_lst=[reinf])
    
    scb = StochasticCB(ccb=cb1)
         
    cbcb = ConstantBondCB(n_z = 500)
        
    rf = SimpleRandomField(n_x = 501,
                           mean = 40,
                           deviation = 6)
    
    print np.amin(rf.field)
        
    ctt = CompositeTensileTest(n_x = 501,
                               L = 500,
                               cb=scb,
                               sig_mu_x= rf.field)
    
    sig_c_i, z_x_i, BC_x_i = ctt.get_cracking_history()
    eps_c_i = ctt.get_eps_c_i(sig_c_i, z_x_i, BC_x_i)
    
        
    load_arr = np.linspace(0, ctt.cb.sig_cu, 100)
    eps_c_arr = ctt.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, load_arr)
    w_dist = ctt.get_w_dist(sig_c_i, z_x_i, BC_x_i)
      
#     ctt.configure_traits()
      
    from matplotlib import pyplot as plt
    plt.subplot(2, 2, 1)
    plt.plot(eps_c_i, sig_c_i)
    plt.plot([0.0, ctt.cb.sig_cu/ctt.cb.E_c], [0.0, ctt.cb.sig_cu])
    plt.plot(eps_c_arr, load_arr)
#     plt.plot([0.0, ctt.cb.eps_fu], [0.0, ctt.cb.sig_cu])
      
#     plt.subplot(2, 2, 2)
#     for i in range(1, len(z_x_i)):
#         plt.plot(ctt.x, ctt.get_eps_f_x(sig_c_i[i], z_x_i[i]))
#         plt.plot(ctt.x, ctt.get_sig_m_x(sig_c_i[i], z_x_i[i]) / ctt.cb.E_m)
#     plt.ylim(ymin=0.0)
    
    plt.subplot(2, 2, 3)
#     plt.plot(ctt.x, z_x_i[-1])
#     plt.plot(ctt.x, ctt.sig_mu_x)
#     plt.plot(ctt.x, ctt.get_sig_m_x(sig_c_i[-2], z_x_i[-2], BC_x_i[-2][0], BC_x_i[-2][1]))
#     plt.hist(w_dist[-2])
    
#     plt.subplot(2, 2, 4)
#     plt.plot(ctt.x, ctt.sig_mu_x)
#     for i in range(1, len(z_x_i)):
#         plt.plot(ctt.x, ctt.get_sig_m_x(sig_c_i[i], z_x_i[i]))
#     plt.ylim(ymin=0.0)

    plt.subplot(2, 2, 2)
    BC_x = BC_x_i[-2]
    plt.plot(ctt.x, ctt.get_sig_m_x(sig_c_i[-2], z_x_i[-2], BC_x[0], BC_x[1]))
    plt.plot(ctt.x, rf.field)
   
    plt.show()

            
    