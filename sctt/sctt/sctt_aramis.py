'''
Created on Jan 20, 2015

@author: Li Yingxiong
'''
from etsproxy.traits.api import \
    HasStrictTraits, Instance, Int, Float, List, Array, Property, \
    cached_property
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq, minimize_scalar, fmin, brute, newton
from .random_fields.simple_random_field import SimpleRandomField
from .crack_bridge_models.constant_bond_cb import ConstantBondCB
from .crack_bridge_models.representative_cb import RepresentativeCB
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
from .crack_bridge_models.random_bond_cb import RandomBondCB
from .reinforcements.fiber_bundle import FiberBundle
import os.path
from scipy.stats import gamma as gam


class CTTAramis(HasStrictTraits):

    cb = EitherType(klasses=[ConstantBondCB,
                             RandomBondCB])  # crack bridge model

    #=========================================================================
    # aramis data
    #=========================================================================
    position = Array
    stress = Array

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

    def gen_data(self):

        z_x_lst = [self.z_x]  # record z array of each cracking state
        # record boundary condition of each cracking state
        BC_x_lst = [self.BC_x]
# sig_c_lst = [0.0] #record cracking load factor

        index = np.argsort(self.stress)
        position = self.position[index]
        stress = self.stress[index]
        for i in range(len(position)):
            self.y.append(position[i])
            z_x_lst.append(np.array(self.z_x))
            BC_x_lst.append(np.array(self.BC_x))
        return np.hstack((0., stress)), np.array(z_x_lst), BC_x_lst

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
#                 sig_m = load*self.cb.E_m/self.cb.E_c*np.ones_like(self.x)y
            else:
                BC_x = BC_x_i[idx]
                eps_arr[i] = np.trapz(self.get_eps_f_x(load, z_x, BC_x[0],
                                                       BC_x[1]), self.x) / self.L
        return eps_arr

    def get_w_dist(self, sig_c_i, z_x_i, BC_x_i):
        '''function for evaluate the crack width
        '''
        w_dist = []
        for sig_c, z_x, BC_x in zip(sig_c_i, z_x_i, BC_x_i):
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

if __name__ == '__main__':

    from .calibration.tau_strength_dependence import interp_tau_shape, interp_tau_scale

    homedir = 'D:\\data\\'
    path = [homedir, 'test61.txt']
    filepath = os.path.join(*path)
    data = np.loadtxt(filepath, delimiter=',')

    sV0_arr = np.array([0.008043056417130334, 0.008689444790342452, 0.009133666555177156,
                        0.00954231413126173, 0.009981034909603366, 0.010372329629894428])
    m_arr = np.array([6., 7., 8., 9., 10., 11.])
    for i, m in enumerate(m_arr):
        sV0 = sV0_arr[i]
        scale = float(interp_tau_scale(sV0, m))
        shape = float(interp_tau_shape(sV0, m))

        reinf1 = ContinuousFibers(r=3.5e-3,
                                  tau=RV(
                                      'gamma', loc=0., scale=scale, shape=shape),
                                  V_f=0.015,
                                  E_f=180e3,
                                  xi=fibers_MC(m=m, sV0=sV0),
                                  label='carbon',
                                  n_int=500)

        cb = RandomBondCB(E_m=25e3,
                          reinforcement_lst=[reinf1],
                          n_BC=12,
                          L_max=400)

        ctta = CTTAramis(n_x=500,
                         L=120,
                         cb=cb,
                         stress=data[3] / 2.,
                         position=data[1])

        sig_c_i, z_x_i, BC_x_i = ctta.gen_data()
        load_arr = np.linspace(0, 20., 200)
        load_arr = np.unique(np.sort(np.hstack((sig_c_i, load_arr))))
        eps_c_arr = ctta.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, load_arr)
        crack_eps_a = ctta.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, sig_c_i)

        plt.plot(eps_c_arr, load_arr, label='m=' + str(m) + ',sV0=' +
                 str(sV0) + ',tau_shape=' + str(shape) + ',tau_scale=' + str(scale))
        plt.plot(crack_eps_a, sig_c_i, 'o')

    plt.plot([0, 20. / reinf1.E_f * 100 / 1.5], [0, 20.])
    plt.xlabel('composite strain')
    plt.ylabel('composite stress [MPa]')

    home_dir = 'D:\\data\\'
#     for i in range(5):
# path1 = [home_dir, 'git',  # the path of the data file
#     'rostar',
#     'scratch',
#     'diss_figs',
#     'TT-6C-0'+str(5)+'.txt']
#     filepath1 = filepath = os.path.join(*path1)
#     data = np.loadtxt(filepath1, delimiter=';')
#     plt.plot(-data[:,2]/2./250. - data[:,3]/2./250.,data[:,1]/2., lw=1, color='0.5')
    path1 = [home_dir, 'stress_strain.txt']
    filepath1 = os.path.join(*path1)
    data = np.loadtxt(filepath1)
    plt.plot(data[0], data[1] / 2, label='aramis')

    home_dir2 = 'D:\\eclipse\\'
    path2 = [home_dir, 'git',  # the path of the data file
             'rostar',
             'scratch',
             'diss_figs',
             'TT-6C-0' + str(5) + '.txt']
    filepath2 = filepath = os.path.join(*path2)
    data = np.loadtxt(filepath2, delimiter=';')
    plt.plot(-data[:, 2] / 2. / 250. - data[:, 3] / 2. / 250.,
             data[:, 1] / 2., lw=1, color='0.5', label='experiment')

    plt.legend(loc='best')
    plt.show()
