from matplotlib import pyplot as plt
import os.path
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy.optimize import brentq, minimize_scalar, fmin, brute, newton
from scipy.stats import gamma as gam
from traits.api import \
    HasStrictTraits, Instance, Int, Float, List, Array, Property, \
    cached_property
from traitsui.api import \
    View, Item, Group
import warnings
from crack_bridge_models.constant_bond_cb import ConstantBondCB
#from crack_bridge_models.random_bond_cb import RandomBondCB
#from crack_bridge_models.representative_cb import RepresentativeCB
import numpy as np
# from quaducom.meso.homogenized_crack_bridge.elastic_matrix.hom_CB_elastic_mtrx \
#    import CompositeCrackBridge
# from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement \
#    import ContinuousFibers
from random_fields.simple_random_field import SimpleRandomField
#from reinforcements.fiber_bundle import FiberBundle
from spirrid.rv import RV
from stats.misc.random_field.random_field_1D import RandomField
from stats.pdistrib.weibull_fibers_composite_distr import \
    WeibullFibers, fibers_MC
import time as t
from util.traits.either_type import EitherType


# from calibration import Calibration
warnings.filterwarnings("error", category=RuntimeWarning)


class CompositeTensileTest(HasStrictTraits):

    cb = Instance(ConstantBondCB)  # crack bridge model

    #=========================================================================
    # Discretization of the specimen
    #=========================================================================
    n_x = Int(501)  # number of material points
    L = Float(1000.)  # the specimen length - mm
    x = Property(depends_on='n_x, L')  # coordinates of the material points

    @cached_property
    def _get_x(self):
        return np.linspace(0, self.L, self.n_x)
    sig_mu_x = Array()

    #=========================================================================
    # Description of cracked specimen
    #=========================================================================
    y = List([])  # the list to record the crack positions
    z_x = Property(depends_on='n_x, L, y')
    '''the array containing the distances from each material point to its 
    nearest crack position
    '''
    @cached_property
    def _get_z_x(self):
        try:
            y = np.array(self.y)
            z_grid = np.abs(self.x[:, np.newaxis] - y[np.newaxis, :])
            return np.amin(z_grid, axis=1)
        except ValueError:  # no cracks exist
            return np.ones_like(self.x) * 2 * self.L

    BC_x = Property(depends_on='x, y')
    '''the array containing the boundary condition for each material point
    '''
    @cached_property
    def _get_BC_x(self):
        try:
            y = np.sort(self.y)
            d = (y[1:] - y[:-1]) / 2.0
            # construct a piecewise function for interpolation
            xp = np.hstack([0, y[:-1] + d, self.x[-1]])
            L_left = np.hstack([y[0], d, np.NAN])
            L_right = np.hstack([d, self.L - y[-1], np.NAN])
            f = interp1d(xp, np.vstack([L_left, L_right]), kind='zero')
            return f(self.x)
        except IndexError:
            return np.vstack([np.zeros_like(self.x), np.zeros_like(self.x)])

    #=========================================================================
    # Strength of the specimen
    #=========================================================================
    experimental_strength = Float(20.)
    strength = Property(depends_on='n_x, L, y, cb')
    '''the strength of the specimen is defined as the minimum of the strength of 
    all crack bridges'''
    @cached_property
    def _get_strength(self):
        if len(self.y) == 0:
            return 1e2
        else:
            y = np.sort(self.y)
            d = (y[1:] - y[:-1]) / 2.0
            L_left = np.hstack([y[0], d])
            L_right = np.hstack([d, self.L - y[-1]])
            v = np.vectorize(self.cb.get_index)
            ind = v(L_left, L_right)
            return np.amin(self.cb.interps[2][ind])
#             return self.experimental_strength

    #=========================================================================
    # Determine the cracking load level
    #=========================================================================
    def get_sig_c_z(self, sig_mu, z, Ll, Lr, sig_c_i_1):
        '''Determine the composite remote stress initiating a crack 
        at position z'''
        fun = lambda sig_c: sig_mu - self.cb.get_sig_m_z(z, Ll, Lr, sig_c)
        try:
            # search the cracking stress level between zero and ultimate
            # composite stress
            return newton(fun, sig_c_i_1)

        except (RuntimeWarning, RuntimeError):
            # no solution, shielded zone
            return 1e6

    get_sig_c_x_i = np.vectorize(get_sig_c_z)

    def get_sig_c_i(self, sig_c_i_1):
        '''Determine the new crack position and level of composite stress
        '''
        # for each material point find the load factor initiating a crack
        sig_c_x_i = self.get_sig_c_x_i(self, self.sig_mu_x,
                                       self.z_x, self.BC_x[0], self.BC_x[1], sig_c_i_1)
        # get the position of the material point corresponding to
        # the minimum cracking load factor
        y_idx = np.argmin(sig_c_x_i)
        y_i = self.x[y_idx]
        sig_c_i = sig_c_x_i[y_idx]
        return sig_c_i, y_i

    #=========================================================================
    # determine the crack history
    #=========================================================================
    def get_cracking_history(self):
        '''Trace the response crack by crack.
        '''
        z_x_lst = [self.z_x]  # record z array of each cracking state
        # record boundary condition of each cracking state
        BC_x_lst = [self.BC_x]
        sig_c_lst = [0.]  # record cracking load factor

        # the first crack initiates at the point of lowest matrix strength
        idx_0 = np.argmin(self.sig_mu_x)
        self.y.append(self.x[idx_0])
        sig_c_0 = self.sig_mu_x[idx_0] * self.cb.E_c / self.cb.E_m
        sig_c_lst.append(sig_c_0)
        print self.sig_mu_x[idx_0], self.x[idx_0]
        z_x_lst.append(np.array(self.z_x))
        BC_x_lst.append(np.array(self.BC_x))

        t1 = t.time()
        print 'begin', t1
        # determine the following cracking load factors
        while True:
            sig_c_i, y_i = self.get_sig_c_i(sig_c_lst[-1])
            if sig_c_i >= self.strength or sig_c_i == 1e6:
                break
            print sig_c_i, y_i
            self.y.append(y_i)
            print 'number of cracks:', len(self.y)
            sig_c_lst.append(sig_c_i)
            z_x_lst.append(np.array(self.z_x))
            BC_x_lst.append(np.array(self.BC_x))
#             self.save_cracking_history(sig_c_i, z_x_lst, BC_x_lst)
#             print 'strength', self.strength
        print 'time consumed', t.time() - t1
        print 'cracking history determined'
        sig_c_u = self.strength
        print sig_c_u
        n_cracks = len(self.y)
        self.y = []
        return np.array(sig_c_lst), np.array(z_x_lst), BC_x_lst, sig_c_u, n_cracks

    #=========================================================================
    # post processing
    #=========================================================================
    def get_eps_c_i(self, sig_c_i, z_x_i, BC_x_i):
        '''For each cracking level calculate the corresponding
        composite strain eps_c.
        '''
        return np.array([np.trapz(self.get_eps_f_x(sig_c, z_x, BC_x[0],
                                                   BC_x[1]), self.x) / self.L
                         for sig_c, z_x, BC_x in zip(sig_c_i, z_x_i, BC_x_i)
                         ])

    def get_eps_f_x(self, sig_c, z_x, Ll_arr, Lr_arr):
        '''function to evaluate specimen reinforcement strain profile
        at given load level and crack distribution
        '''
        eps_f = np.zeros_like(self.x)
        z_x_map = np.argsort(z_x)
        eps_f[z_x_map] = self.cb.get_eps_f_z(
            z_x[z_x_map], Ll_arr, Lr_arr, sig_c)
        return eps_f

    def get_sig_m_x(self, sig_c, z_x, Ll_arr, Lr_arr):
        '''function to evaluate specimen matrix stress profile
        at given load level and crack distribution
        '''
        eps_m = np.zeros_like(self.x)
        z_x_map = np.argsort(z_x)
        eps_m[z_x_map] = self.cb.get_sig_m_z(z_x[z_x_map], Ll_arr[z_x_map],
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
            if np.any(z_x == 2 * self.L):
                eps_arr[i] = load / self.cb.E_c
                sig_m = self.cb.E_m * eps_arr[i] * np.ones_like(self.x)
            else:
                BC_x = BC_x_i[idx]
                eps_arr[i] = np.trapz(self.get_eps_f_x(load, z_x, BC_x[0],
                                                       BC_x[1]), self.x) / self.L
                sig_m = self.get_sig_m_x(load, z_x, BC_x[0], BC_x[1])
            # save the cracking history
#             plt.clf()
#             plt.plot(self.x, sig_m)
#             plt.plot(self.x, self.sig_mu_x)
#             plt.ylim((0., 1.2 * np.max(self.sig_mu_x)))
#             savepath = 'D:\cracking history\\1\\load_step' + \
#                 str(i + 1) + '.png'
#             plt.savefig(savepath)

        return eps_arr

#     def get_w_dist(self, sig_c_i, z_x_i, BC_x_i, load_arr):
#         '''function for evaluate the crack width
#         '''
#         w_dist = []
#         for sig_c, z_x, BC_x in zip(sig_c_i, z_x_i, BC_x_i):
#             eps_f_x = self.get_eps_f_x(sig_c, z_x, BC_x[0], BC_x[1])
#             eps_m_x = self.get_sig_m_x(sig_c, z_x, BC_x[0], BC_x[1]) / self.cb.E_m
#             y = self.x[z_x == 0]
#             if not y.size:
#                 continue
#             distance = np.abs(self.x[:, np.newaxis] - y[np.newaxis, :])
#             nearest_crack = y[np.argmin(distance, axis=1)]
#             w_arr = np.array([np.trapz( (eps_f_x[nearest_crack == y_i] - \
#                                          eps_m_x[nearest_crack == y_i]), \
#                                        self.x[nearest_crack == y_i] ) \
#                                for y_i in y])
#             w_dist.append(w_arr)
#         return w_dist

    def get_w_dist(self, sig_c_i, z_x_i, BC_x_i, load_arr):
        '''function for evaluate the crack width
        '''
        w_dist = []
        for sig_c in load_arr:
            idx = np.searchsorted(sig_c_i, sig_c) - 1
            z_x = z_x_i[idx]
            if np.any(z_x == 2 * self.L):
                w_arr = np.array([np.nan])
            else:
                BC_x = BC_x_i[idx]
                eps_f_x = self.get_eps_f_x(sig_c, z_x, BC_x[0], BC_x[1])
                eps_m_x = self.get_sig_m_x(
                    sig_c, z_x, BC_x[0], BC_x[1]) / self.cb.E_m
                y = self.x[z_x == 0]
                if not y.size:
                    continue
                distance = np.abs(self.x[:, np.newaxis] - y[np.newaxis, :])
                nearest_crack = y[np.argmin(distance, axis=1)]
                w_arr = np.array([np.trapz((eps_f_x[nearest_crack == y_i] -
                                            eps_m_x[nearest_crack == y_i]),
                                           self.x[nearest_crack == y_i])
                                  for y_i in y])
            w_dist.append(w_arr)

        return w_dist

    def get_crack_opening(self, z_x, BC_x, load):
        '''evaluate the crack openings for gving load(single)'''
        eps_f_x = self.get_eps_f_x(load, z_x, BC_x[0], BC_x[1])
        eps_m_x = self.get_sig_m_x(load, z_x, BC_x[0], BC_x[1]) / self.cb.E_m
        y = self.x[z_x == 0]
        distance = np.abs(self.x[:, np.newaxis] - y[np.newaxis, :])
        nearest_crack = y[np.argmin(distance, axis=1)]
        w_arr = np.array([np.trapz((eps_f_x[nearest_crack == y_i] -
                                    eps_m_x[nearest_crack == y_i]),
                                   self.x[nearest_crack == y_i])
                          for y_i in y])
        return w_arr

    def get_damage_arr(self, sig_c_i, z_x_i, BC_x_i, load_arr):
        '''function to evaluate the damage probability of reinforcement corresponding 
        to the given load_arr
        '''
        damage_arr = np.ones_like(load_arr)
        for i, load in enumerate(load_arr + 1e-10):
            idx = np.searchsorted(sig_c_i, load) - 1
            z_x = z_x_i[idx]
            if np.any(z_x == 2 * self.L):
                damage_arr[i] = 0
            else:
                y = self.x[z_x == 0]
                d = (y[1:] - y[:-1]) / 2.0
                L_left = np.hstack([y[0], d])
                L_right = np.hstack([d, self.L - y[-1]])
                v = np.vectorize(self.cb.get_index)
                ind = v(L_left, L_right)
                damage = []
                for index in ind:
                    interp_damage = self.cb.interps[3][index]
                    damage.append(interp_damage(load))
                damage_arr[i] = np.amax(damage)
        return damage_arr

    def save_cracking_history(self, sig_c_i, z_x_lst, BC_x_lst):
        '''save the cracking history'''
        plt.clf()
        plt.subplot(411)
        i = len(z_x_lst)
        BC_x = BC_x_lst[i - 2]
        sig_m = self.get_sig_m_x(sig_c_i, z_x_lst[i - 2], BC_x[0], BC_x[1])
        plt.plot(self.x, sig_m)
        plt.plot(self.x, self.sig_mu_x)

        plt.subplot(412)
        plt.plot(self.x, z_x_lst[i - 2])

        plt.subplot(413)
        plt.plot(self.x, BC_x[0])

        plt.subplot(414)
        plt.plot(self.x, BC_x[1])
        savepath = 'D:\cracking history\\1\\BC' + str(len(self.y) - 1) + '.png'
        plt.savefig(savepath)

    def get_d_m(self, sig_c_i, z_x_i, BC_x_i, load_arr):
        '''evaluate the matrix displacement'''
        d_m_arr = np.zeros((len(load_arr), self.n_x))
        for i, load in enumerate(load_arr):
            idx = np.searchsorted(sig_c_i, load) - 1
            z_x = z_x_i[idx]
            if np.any(z_x == 2 * self.L):
                d_m_arr[i] = load / self.cb.E_c * self.x
            else:
                BC_x = BC_x_i[idx]
                eps_m_x = self.get_sig_m_x(
                    load, z_x, BC_x[0], BC_x[1]) / self.cb.E_m
                # displacement purely contributed by the matrix strain
                d_m = np.hstack((0., cumtrapz(eps_m_x, self.x)))
                # add the crack opening
                w = self.get_crack_opening(z_x, BC_x, load)
                crack_idx = np.where(z_x == 0)[0]
                for j, idx in enumerate(crack_idx):
                    d_m[idx:] = d_m[idx:] + w[j]
            d_m_arr[i] = d_m

            plt.clf()
            plt.plot(self.x, d_m)
            savepath = 'D:\cracking history\\1\\load_step' + \
                str(i + 1) + '.png'
            plt.savefig(savepath)

        return d_m_arr

    view = View(Item('L', show_label=False),
                Item('n_x', show_label=False),
                Item('cb', show_label=False),
                buttons=['OK', 'Cancel'])

#=============================================================================
# output
#=============================================================================
if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    home_dir = 'D:\\Eclipse\\'
    home_dir = '/home/rch'
    for i in range(5):
        #     for i in [2]:
        path1 = [home_dir, 'git',  # the path of the data file
                 'rostar',
                 'scratch',
                 'diss_figs',
                 'TT-4C-0' + str(i + 1) + '.txt']
        filepath1 = filepath = os.path.join(*path1)

        path2 = [home_dir, 'git',  # the path of the data file
                 'rostar',
                 'scratch',
                 'diss_figs',
                 'TT-6C-0' + str(i + 1) + '.txt']
        filepath2 = os.path.join(*path2)

        data = np.loadtxt(filepath1, delimiter=';')
        plt.plot(-data[:, 2] / 2. / 250. - data[:, 3] / 2. / 250.,
                 data[:, 1] / 2., lw=1, color='0.5')
        data = np.loadtxt(filepath2, delimiter=';')
        plt.plot(-data[:, 2] / 2. / 250. - data[:, 3] / 2. / 250.,
                 data[:, 1] / 2., lw=1, color='0.5')

    from calibration.tau_strength_dependence import interp_tau_shape, interp_tau_scale
    from calibration.matrix_strength_dependence import interp_m_shape

    s_arr = np.array(
        [0.00923835,  0.00979774,  0.01029609,  0.01075946,  0.01117243, 0.01154383])

    s = s_arr[0]
    m = 6.

    shape = interp_tau_shape(s, m)
    scale = interp_tau_scale(s, m)

    print shape, scale

    cb = ConstantBondCB()

    random_field = RandomField(seed=False,
                               lacor=1.,
                               length=500,
                               nx=1000,
                               nsim=1,
                               loc=.0,
                               shape=45.,
                               scale=3.599,
                               distr_type='Weibull')

    ctt = CompositeTensileTest(n_x=1000,
                               L=500,
                               cb=cb,
                               sig_mu_x=random_field.random_field)

    sig_c_i, z_x_i, BC_x_i, sig_c_u, n_crack = ctt.get_cracking_history()
#     eps_c_i = ctt.get_eps_c_i(sig_c_i, z_x_i, BC_x_i)
    print '1.0%', [sig_c_i]
    print np.sort(ctt.y)

    load_arr = np.unique(np.hstack((np.linspace(0, 30, 100), sig_c_i)))
    eps_c_arr = ctt.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, load_arr)
#     damage_arr = ctt.get_damage_arr(sig_c_i, z_x_i, BC_x_i, load_arr)
    crack_eps = ctt.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, sig_c_i)

    plt.plot(eps_c_arr, load_arr, 'k', lw=2, label='$v_\mathrm{f}=1.0\%$')
    plt.plot(crack_eps, sig_c_i, 'ko')

    plt.legend(loc='best')
    plt.show()
