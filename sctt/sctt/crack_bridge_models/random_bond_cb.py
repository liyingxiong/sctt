from enthought.traits.api import HasTraits, Array, Instance, List, Float, Int, \
    Property, cached_property
from util.traits.either_type import EitherType
from types import FloatType
from spirrid.rv import RV
from stats.pdistrib.weibull_fibers_composite_distr import WeibullFibers
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement import \
    ContinuousFibers
from reinforcements.fiber_bundle import FiberBundle
import numpy as np
from scipy.optimize import root, brentq, fminbound, brute, minimize, fmin_cg
from scipy.integrate import cumtrapz
import time as t
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d, interp2d, griddata, LinearNDInterpolator

        
class RandomBondCB(HasTraits):
    
    #===============================================================================
    # Material Parameters
    #===============================================================================    
    reinforcement_lst = List(EitherType(klasses=[FiberBundle, ContinuousFibers]))
    E_m = Float(25e3) # the elastic modulus of the matrix
    w = Float # the crack width
    # the BCs
    Lr = Float
    Ll = Float
    
    V_f_tot = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_V_f_tot(self):
        V_f_tot = 0.0
        for reinf in self.reinforcement_lst:
            V_f_tot += reinf.V_f
        return V_f_tot
    
    E_c = Property(depends_on='reinforcement_lst+, E_m')
    @cached_property
    def _get_E_c(self):
        E_fibers = 0.0
        for reinf in self.reinforcement_lst:
            E_fibers += reinf.V_f * reinf.E_f
        E_c = self.E_m * (1. - self.V_f_tot) + E_fibers
        return E_c * (1. + 1e-15)
    
    #===============================================================================
    # sort the integral points according to bond intensity in descending order
    #===============================================================================    
    sorted_theta = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_sorted_theta(self):
        depsf_arr = np.array([]) #bond intensity, 2*tau/r
        V_f_arr = np.array([])
        E_f_arr = np.array([])
        xi_arr = np.array([])
        stat_weights_arr = np.array([])
        nu_r_arr = np.array([])
        r_arr = np.array([])
        for reinf in self.reinforcement_lst:
            n_int = len(np.hstack((np.array([]), reinf.depsf_arr)))
            depsf_arr = np.hstack((depsf_arr, reinf.depsf_arr))
            V_f_arr = np.hstack((V_f_arr, np.repeat(reinf.V_f, n_int)))
            E_f_arr = np.hstack((E_f_arr, np.repeat(reinf.E_f, n_int)))
            xi_arr = np.hstack((xi_arr, np.repeat(reinf.xi, n_int)))
            if isinstance(reinf.tau, np.ndarray):
                stat_weights_arr = np.hstack((stat_weights_arr, reinf.stat_weights))
            else:
                stat_weights_arr = np.hstack((stat_weights_arr,
                                              np.repeat(reinf.stat_weights, n_int)))
            nu_r_arr = np.hstack((nu_r_arr, reinf.nu_r))
            r_arr = np.hstack((r_arr, reinf.r_arr))
        argsort = np.argsort(depsf_arr)[::-1]
        # sorting the masks for the evaluation of F
        idxs = np.array([])
        for i, reinf in enumerate(self.reinforcement_lst):
            idxs = np.hstack((idxs, i * np.ones_like(reinf.depsf_arr)))
        masks = []
        for i, reinf in enumerate(self.reinforcement_lst):
            masks.append((idxs == i)[argsort])
        max_depsf = [np.max(reinf.depsf_arr) for reinf in self.reinforcement_lst]
        masks = [masks[i] for i in np.argsort(max_depsf)[::-1]]
        return depsf_arr[argsort], V_f_arr[argsort], E_f_arr[argsort], \
                xi_arr[argsort], stat_weights_arr[argsort], \
                nu_r_arr[argsort], masks, r_arr[argsort]

    sorted_depsf = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_sorted_depsf(self):
        return self.sorted_theta[0]

    sorted_V_f = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_sorted_V_f(self):
        return self.sorted_theta[1]

    sorted_E_f = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_sorted_E_f(self):
        return self.sorted_theta[2]

    sorted_xi = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_sorted_xi(self):
        return self.sorted_theta[3]

    sorted_stats_weights = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_sorted_stats_weights(self):
        return self.sorted_theta[4]

    sorted_nu_r = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_sorted_nu_r(self):
        return self.sorted_theta[5]

    sorted_masks = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_sorted_masks(self):
        return self.sorted_theta[6]

    sorted_r = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_sorted_r(self):
        return self.sorted_theta[7]

    sorted_xi_cdf = Property(depends_on='reinforcement_lst+,Ll,Lr')
    @cached_property
    def _get_sorted_xi_cdf(self):
        '''breaking strain: CDF for random and Heaviside for discrete values'''
#         TODO: does not work for reinforcement types with the same xi: line 156
        methods = []
        masks = [] 
        for reinf in self.reinforcement_lst:
#             print self.sorted_xi
#             print reinf.xi
            masks.append(self.sorted_xi == reinf.xi)
            if isinstance(reinf.xi, FloatType):
                methods.append(lambda x: 1.0 * (reinf.xi <= x))
            elif isinstance(reinf.xi, RV):
                methods.append(reinf.xi._distr.cdf)
            elif isinstance(reinf.xi, WeibullFibers):
                reinf.xi.Ll = self.Ll
                reinf.xi.Lr = self.Lr
                methods.append(reinf.xi.cdf)
        return methods, masks
    
    Kf = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_Kf(self):
        return self.sorted_V_f * self.sorted_nu_r * \
                self.sorted_stats_weights * self.sorted_E_f


    #===============================================================================
    # evaluate maximum fiber strain and matrix strain profile 
    #===============================================================================        
    def dem_depsf_vect(self, damage):
        '''evaluates the deps_m given deps_f at that point and the damage array
        Eq.(4.17)
        '''
        Kf_intact = self.Kf * (1. - damage)
        Kf_intact_bonded = np.hstack((0.0, np.cumsum((Kf_intact))))[:-1]
        Kf_broken = np.sum(self.Kf - Kf_intact)
        Kf_add = Kf_intact_bonded + Kf_broken
        Km = (1. - self.V_f_tot) * self.E_m
        E_mtrx = Km + Kf_add
        mu_T = np.cumsum((self.sorted_depsf * Kf_intact)[::-1])[::-1]
        return mu_T / E_mtrx
        
    def F(self, dems, amin):
        '''Auxiliary function (see Part II, appendix B), Eq.(D.21) and Eq.(D.35)
        '''
        F = np.zeros_like(self.sorted_depsf)
        for i, mask in enumerate(self.sorted_masks):
            depsfi = self.sorted_depsf[mask]
            demsi = dems[mask]
            fi = 1. / (depsfi + demsi)
            F[mask] = np.hstack((np.array([0.0]), cumtrapz(fi, -depsfi)))
            if i == 0:
                C = 0.0
            else:
                depsf0 = self.sorted_depsf[self.sorted_masks[i - 1]]
                depsf1 = depsfi[0]
                idx = np.sum(depsf0 > depsf1) - 1
                depsf2 = depsf0[idx]
                a1 = np.exp(F[self.sorted_masks[i - 1]][idx] / 2. + np.log(amin))
                p = depsf2 - depsf1
                q = depsf1 + demsi[0]
                amin_i = np.sqrt(a1 ** 2 + p / q * a1 ** 2)
                C = np.log(amin_i / amin)
            F[mask] += 2 * C
        return F
    
    def clamped(self, Lmin, Lmax, init_dem):
        '''maximum fiber strain and matrix strain profile for clamped fiber
        '''
        a = np.hstack((-Lmin, 0.0, Lmax))
        em = np.hstack((init_dem * Lmin, 0.0, init_dem * Lmax))
        epsf0 = (self.sorted_depsf / 2. * (Lmin ** 2 + Lmax ** 2) +
                     self.w + em[0] * Lmin / 2. + em[-1] * Lmax / 2.) / (Lmin + Lmax)
        return a, em, epsf0

    _x_arr = Array
    def __x_arr_default(self):
        return np.repeat(1e-10, len(self.sorted_depsf))

    _epsm_arr = Array
    def __epsm_arr_default(self):
        return np.repeat(1e-10, len(self.sorted_depsf))

    _epsf0_arr = Array
    def __epsf0_arr_default(self):
        return np.repeat(1e-10, len(self.sorted_depsf))
    
    def profile(self, iter_damage, Lmin, Lmax):
        '''evaluate maximum fiber strain and matrix strain profile according to given 
        damage portion and boundary condition
        '''
        # matrix strain derivative with resp. to z as a function of T
        dems = self.dem_depsf_vect(iter_damage)
        # initial matrix strain derivative
        init_dem = dems[0]
        # debonded length of fibers with Tmax
        amin = (self.w / (np.abs(init_dem) + np.abs(self.sorted_depsf[0]))) ** 0.5
        # integrated f(depsf) - see article
        F = self.F(dems, amin)
        # a1 is a(depsf) for double sided pullout
        a1 = amin * np.exp(F / 2.)
        aX = np.exp((-np.log(np.abs(self.sorted_depsf) + dems) + np.log(self.w)) / 2.)
        if Lmin < a1[0] and Lmax < a1[0]:
            # all fibers debonded up to Lmin and Lmax
            a, em, epsf0 = self.clamped(Lmin, Lmax, init_dem)

        elif Lmin < a1[0] and Lmax >= a1[0]:
            # all fibers debonded up to Lmin but not up to Lmax
            amin = -Lmin + np.sqrt(2 * Lmin ** 2 + 2 * self.w / (self.sorted_depsf[0] + init_dem))
            C = np.log(amin ** 2 + 2 * Lmin * amin - Lmin ** 2)
            a2 = np.sqrt(2 * Lmin ** 2 + np.exp((F + C))) - Lmin
            if Lmax < a2[0]:
                a, em, epsf0 = self.clamped(Lmin, Lmax, init_dem)
            else:
                if Lmax <= a2[-1]:
                    idx = np.sum(a2 < Lmax) - 1
                    a = np.hstack((-Lmin, 0.0, a2[:idx + 1], Lmax))
                    em2 = np.cumsum(np.diff(np.hstack((0.0, a2))) * dems)
                    em = np.hstack((init_dem * Lmin, 0.0, em2[:idx + 1], em2[idx] + (Lmax - a2[idx]) * dems[idx]))
                    um = np.trapz(em, a)
                    epsf01 = em2[:idx + 1] + a2[:idx + 1] * self.sorted_depsf[:idx + 1]
                    epsf02 = (self.w + um + self.sorted_depsf[idx + 1:] / 2. * (Lmin ** 2 + Lmax ** 2)) / (Lmin + Lmax)
                    epsf0 = np.hstack((epsf01, epsf02))
                else:
                    a = np.hstack((-Lmin, 0.0, a2, Lmax))
                    em2 = np.cumsum(np.diff(np.hstack((0.0, a2))) * dems)
                    em = np.hstack((init_dem * Lmin, 0.0, em2, em2[-1]))
                    epsf0 = em2 + self.sorted_depsf * a2
        elif a1[0] < Lmin and a1[-1] > Lmin:
            # some fibers are debonded up to Lmin, some are not
            # boundary condition position
            idx1 = np.sum(a1 <= Lmin)
            # a(T) for one sided pullout
            # first debonded length amin for one sided PO
            depsfLmin = self.sorted_depsf[idx1]
            p = (depsfLmin + dems[idx1])
            a_short = np.hstack((a1[:idx1], Lmin))
            em_short = np.cumsum(np.diff(np.hstack((0.0, a_short))) * dems[:idx1 + 1])
            emLmin = em_short[-1]
            umLmin = np.trapz(np.hstack((0.0, em_short)), np.hstack((0.0, a_short)))
            amin = -Lmin + np.sqrt(4 * Lmin ** 2 * p ** 2 - 4 * p * emLmin * Lmin + 4 * p * umLmin - 2 * p * Lmin ** 2 * depsfLmin + 2 * p * self.w) / p
            C = np.log(amin ** 2 + 2 * amin * Lmin - Lmin ** 2)
            a2 = (np.sqrt(2 * Lmin ** 2 + np.exp(F + C - F[idx1])) - Lmin)[idx1:]
            # matrix strain profiles - shorter side
            a_short = np.hstack((-Lmin, -a1[:idx1][::-1], 0.0))
            dems_short = np.hstack((dems[:idx1], dems[idx1]))
            em_short = np.hstack((0.0, np.cumsum(np.diff(-a_short[::-1]) * dems_short)))[::-1]
            if a2[-1] > Lmax:
                idx2 = np.sum(a2 <= Lmax)
                # matrix strain profiles - longer side
                a_long = np.hstack((a1[:idx1], a2[:idx2]))
                em_long = np.cumsum(np.diff(np.hstack((0.0, a_long))) * dems[:idx1 + idx2])
                a = np.hstack((a_short, a_long, Lmax))
                em = np.hstack((em_short, em_long, em_long[-1] + (Lmax - a_long[-1]) * dems[idx1 + idx2]))
                um = np.trapz(em, a)
                epsf01 = em_long + a_long * self.sorted_depsf[:idx1 + idx2]
                epsf02 = (self.w + um + self.sorted_depsf [idx1 + idx2:] / 2. * (Lmin ** 2 + Lmax ** 2)) / (Lmin + Lmax)
                epsf0 = np.hstack((epsf01, epsf02))
            else:
                a_long = np.hstack((0.0, a1[:idx1], a2, Lmax))
                a = np.hstack((a_short, a_long[1:]))
                dems_long = dems
                em_long = np.hstack((np.cumsum(np.diff(a_long[:-1]) * dems_long)))
                em_long = np.hstack((em_long, em_long[-1]))
                em = np.hstack((em_short, em_long))
                epsf0 = em_long[:-1] + self.sorted_depsf * a_long[1:-1]
        elif a1[-1] <= Lmin:
            # free debonding
            a = np.hstack((-Lmin, -a1[::-1], 0.0, a1, Lmax))
            em1 = np.cumsum(np.diff(np.hstack((0.0, a1))) * dems)
            em = np.hstack((em1[-1], em1[::-1], 0.0, em1, em1[-1]))
            epsf0 = em1 + self.sorted_depsf * a1
        self._x_arr = a
        self._epsm_arr = em
        self._epsf0_arr = epsf0
        a_short = -a[a < 0.0][1:][::-1]
        if len(a_short) < len(self.sorted_depsf):
            a_short = np.hstack((a_short, Lmin * np.ones(len(self.sorted_depsf) - len(a_short))))
        a_long = a[a > 0.0][:-1]
        if len(a_long) < len(self.sorted_depsf):
            a_long = np.hstack((a_long, Lmax * np.ones(len(self.sorted_depsf) - len(a_long))))
        return epsf0, a_short, a_long

    #===============================================================================
    # calculate the damage portion of the reinforcement
    #===============================================================================        
    def vect_xi_cdf(self, epsy, x_short, x_long):
        '''evaluate the damage portion for the given maximum fiber strain array
        and boundary condition
        '''
        Pf = np.zeros_like(self.sorted_depsf)
        methods, masks = self.sorted_xi_cdf
        for i, method in enumerate(methods):
            if method.__doc__ == 'weibull_fibers_cdf_mc':
                Pf[masks[i]] += method(epsy[masks[i]],
                                       self.sorted_depsf[masks[i]],
                                       self.sorted_r[masks[i]],
                                       x_short[masks[i]],
                                       x_long[masks[i]])
            elif method.__doc__ == 'weibull_fibers_cdf_cb_elast':
                Pf[masks[i]] += method(epsy[masks[i]],
                                       self.sorted_depsf[masks[i]],
                                       self.sorted_r[masks[i]],
                                       x_short[masks[i]],
                                       x_long[masks[i]])
            else:
                Pf[masks[i]] += method(epsy[masks[i]])
        return Pf

    def damage_residuum(self, iter_damage):
        if np.any(iter_damage < 0.0) or np.any(iter_damage > 1.0):
            return np.ones_like(iter_damage) * 2.0
        else:
            Lmin = min(self.Ll, self.Lr)
            Lmax = max(self.Ll, self.Lr)
            epsf0, x_short, x_long = self.profile(iter_damage, Lmin, Lmax)
            residuum = self.vect_xi_cdf(epsf0, x_short=x_short, x_long=x_long) - iter_damage
            return residuum
        
    damage = Property(depends_on='w, Ll, Lr, reinforcement+')
    @cached_property
    def _get_damage(self):
        if self.w == 0.:
            damage = np.zeros_like(self.sorted_depsf)
        else:
            try:
                damage = root(self.damage_residuum, np.ones_like(self.sorted_depsf) * 0.2,
                              method='excitingmixing', options={'maxiter':100})
                if np.any(damage.x < 0.0) or np.any(damage.x > 1.0):
                    raise ValueError
                damage = damage.x
            except:
                print 'fast opt method does not converge: switched to a slower, robust method for this step'
                damage = root(self.damage_residuum, np.ones_like(self.sorted_depsf) * 0.2,
                              method='hybr')
                damage = damage.x
        return damage
    
    def sig_c(self, w): #for evaluation of maximum w
        self.w = float(w)
        self.damage
        sig_c = np.sum(self._epsf0_arr \
                       *self.sorted_stats_weights \
                       *self.sorted_V_f*self.sorted_nu_r \
                       *self.sorted_E_f*(1. - self.damage))
        return sig_c

    
#     #===============================================================================
#     # composite stress at crack position as a function of crack width w
#     #===============================================================================
#     def sig_c(self, w):
#         self.w = float(w)
#         self.damage
#         sig_c = np.sum(self._epsf0_arr*self.sorted_stats_weights \
#                        *self.sorted_V_f*self.sorted_nu_r \
#                        *self.sorted_E_f*(1. - self.damage))
#         return sig_c
#     
#     #===============================================================================
#     # maximum sig_c and the corresponding crack width w_max
#     #===============================================================================
#     def sig_c_max(self):
#         def minfunc_sig_c(w):
#             self.w = float(w)
#             self.damage
#             stiffness_loss = np.sum(self.Kf * self.damage) / np.sum(self.Kf)
#             if stiffness_loss > 0.90:
#                 return 1. + w
#             return -self.sig_c(w)
#         
#         def residuum_stiffness(w):
#             self.w = w
#             self.damage
#             stiffness_loss = np.sum(self.Kf * self.damage) / np.sum(self.Kf)
#             if stiffness_loss > 0.90:
#                 return 1. + w
#             if stiffness_loss < 0.65 and stiffness_loss > 0.45:
#                 residuum = 0.0
#             else:
#                 residuum = stiffness_loss - 0.5
#             return residuum        
#         
#         w_max = brentq(residuum_stiffness, 0.0, min(0.1 * (self.Ll + self.Lr), 20.))
#         w_points = np.linspace(0, w_max, 7)
#         w_maxima = []
#         sigma_maxima = []
#         for i, w in enumerate(w_points[1:]):
#             w_max = fminbound(minfunc_sig_c, w_points[i], w_points[i + 1], maxfun=5, disp=0)
#             w_maxima.append(w_max)
#             sigma_maxima.append(self.sig_c(w_max))
#         return sigma_maxima[np.argmax(np.array(sigma_maxima))], w_maxima[np.argmax(np.array(sigma_maxima))]
# 
#     #===============================================================================
#     # interpolation parameters
#     #===============================================================================  
#     n_BC = Int(8) #number of different boundary conditions in the interpolator
#     
#     n_Int = Property(depends_on='reinforcement_lst')
#     @cached_property
#     def _get_n_Int(self):
#         n_Int = 0
#         for reinf in self.reinforcement_lst:
#             n_Int += reinf.n_int
#         return n_Int
# 
#     #===============================================================================
#     # discretization of the boundary condition
#     #===============================================================================
#     BC_range = Property(depends_on = 'reinforcement_lst+, E_m')
#     @cached_property
#     def _get_BC_range(self):
#         self.Lr, self.Ll = 1e5, 1e5
#         self.w = self.sig_c_max()[1]
#         self.damage
#         L_max = self._x_arr[-2] #maximum debonding length
#         BC_range = np.logspace(np.log10(0.02*L_max), np.log10(L_max), self.n_BC)
#         return BC_range
#     
#     #=============================================================================
#     # prepare the interpolator for each Boundary Condition
#     #=============================================================================    
#     interps = Property(denpends_on='reinforcement_lst+, Ll, Lr, n_BC, E_m')
#     @cached_property
#     def _get_interps(self):
#         interps_sigm = []
#         interps_epsf = []
#         sig_max_lst = []
#         t1 = t.clock()
#         print 'preparing the interpolators:'
#         for j, L_r in enumerate(self.BC_range):
#             for q, L_l in enumerate(self.BC_range):
#                 if L_l <= L_r:
#                     self.Ll = L_l
#                     self.Lr = L_r
#                     sigma_max, w_max = self.sig_c_max()
#                     # w is discretized in such a way that the corresponding sig_c 
#                     # are more evenly distributed
#                     w_arr = np.linspace(np.sqrt(w_max), 1e-15, 20)**2
#                     # a uniform x coordinates for interpolation 
#                     x_arr_record = np.linspace(0, L_r, self.n_Int)
#      
#                     epsf_record = []
#                     sigm_record = []
#                     sigc_record = []
#                     
#                     for w in w_arr:
#                         self.w = w
#                         self.damage
#                                                                         
#                         epsf_x = np.zeros_like(self._x_arr)
#                         for i, depsf in enumerate(self.sorted_depsf):
#                             epsf_x += np.maximum(self._epsf0_arr[i] - \
#                                 depsf * np.abs(self._x_arr), \
#                                 self._epsm_arr)*self.sorted_stats_weights[i]
#                                 
#                         sigm_x = self._epsm_arr*self.E_m
#                             
#                         sigc = np.sum(self._epsf0_arr*self.sorted_stats_weights \
#                                     *self.sorted_V_f*self.sorted_nu_r \
#                                     *self.sorted_E_f*(1. - self.damage))
#                         
#                         #=============================================================================
#                         # reshape the strain and stress array to make the data on a regular grid
#                         epsf_x = griddata(self._x_arr, epsf_x, x_arr_record)
#                         sigm_x = griddata(self._x_arr, sigm_x, x_arr_record)
#                         #=============================================================================
#                                
#                         epsf_record.append(epsf_x)
#                         sigm_record.append(sigm_x)
#                         sigc_record.append(sigc)
#                         
#                     #=============================================================================
#                     # plot the stress or strain profile under given BC                     
# #                     if L_l == self.BC_range[-1] and L_r == self.BC_range[-1]:
# #                         X, Y = np.meshgrid(x_arr_record, sigc_record)
# #                         fig = plt.figure()
# #                         ax = fig.add_subplot(111, projection='3d')
# #                         ax.plot_wireframe(X, Y, sigm_record, rstride=2, cstride=20)
#                     #=============================================================================
#   
#                     interp_epsf = interp2d(x_arr_record, sigc_record, epsf_record)
#                     interp_sigm = interp2d(x_arr_record, sigc_record, sigm_record)
#                     
#                     print ((j+1)*j/2+q+1)*100/(self.n_BC*(self.n_BC+1)/2), '%'
#                     
#                     interps_epsf.append(interp_epsf)
#                     interps_sigm.append(interp_sigm)
#                     sig_max_lst.append(sigma_max)
#                     
#         print 'time consumed:', t.clock()-t1
#         return interps_epsf, interps_sigm, sig_max_lst
# 
#     #=============================================================================
#     # functions for evaluation of stress and strain profile 
#     #=============================================================================    
#     def get_index(self, Ll, Lr):
#         # find the index of the interpolator corresponding to the BC
#         l, r = np.sort([Ll, Lr])
#         i = (np.abs(self.BC_range - l)).argmin()
#         j = (np.abs(self.BC_range - r)).argmin()
#         return (j+1)*j/2+i
#     
#     # function for evaluating specimen reinforcement strain    
#     def get_eps_f_z(self, z_arr, Ll_arr, Lr_arr, load):
#         def get_eps_f_i(self, z, Ll, Lr, load):
#             ind = self.get_index(Ll, Lr)
#             f = self.interps[0][ind]
#             return f(z, load)        
#         v = np.vectorize(get_eps_f_i)
#         return v(self, z_arr, Ll_arr, Lr_arr, load)
# 
#     # function for evaluating specimen matrix stress       
#     def get_sig_m_z(self, z_arr, Ll_arr, Lr_arr, load):
#         def get_sig_m_i(self, z, Ll, Lr, load):
#             ind = self.get_index(Ll, Lr)
#             f = self.interps[1][ind]
#             return f(z, load)
#         v = np.vectorize(get_sig_m_i)
#         return v(self, z_arr, Ll_arr, Lr_arr, load)

if __name__ == '__main__':
    
    reinf = FiberBundle(r=0.0035,
                      tau=np.array([ 0.48419826,  0.71887589,  0.86655175,  0.98916772,  1.10160428, 1.21166999,  1.32595141,  1.45322794,  1.61204695,  1.89332916]),
                      tau_weights = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
                      V_f=0.1,
                      E_f=240e3,
                      xi=0.035)
#     from calibration import Calibration
#     import os.path
#     from stats.pdistrib.weibull_fibers_composite_distr import \
#         WeibullFibers, fibers_MC
# 
#     
#     w_arr = np.linspace(0.0, np.sqrt(8.), 401) ** 2    
#     home_dir = 'D:\\Eclipse\\'
#     path = [home_dir, 'git',  # the path of the data file
#             'rostar',
#             'scratch',
#             'diss_figs',
#             'CB1.txt']
#     filepath = os.path.join(*path)
#     exp_data = np.zeros_like(w_arr)
#     file1 = open(filepath, 'r')
#     cb = np.loadtxt(file1, delimiter=';')
#     test_xdata = -cb[:, 2] / 4. - cb[:, 3] / 4. - cb[:, 4] / 2.
#     test_ydata = cb[:, 1] / (11. * 0.445) * 1000
#     interp = interp1d(test_xdata, test_ydata, bounds_error=False, fill_value=0.)
#     exp_data = interp(w_arr)
#     cali = Calibration(experi_data=exp_data,
#                        w_arr=w_arr,
#                        tau_arr=np.logspace(np.log10(1e-5), np.log10(1), 200))
#     tau_ind = np.nonzero(cali.tau_weights)
#     
#     reinf = FiberBundle(r=0.0035,
#                   tau=cali.tau_arr[tau_ind],
#                   tau_weights = cali.tau_weights[tau_ind],
#                   V_f=0.01,
#                   E_f=200e3,
#                   xi=fibers_MC(m=cali.m, sV0=cali.sV0))





    from stats.pdistrib.weibull_fibers_composite_distr import WeibullFibers, fibers_MC
#     reinf = ContinuousFibers(r=3.5e-3,
#                               tau=RV('weibull_min', loc=0.01, scale=.1, shape=2.),
#                               V_f=0.005,
#                               E_f=200e3,
#                               xi=fibers_MC(m=7., sV0=0.005),
#                               label='carbon',
#                               n_int=200)

#     ccb = RandomBondCB(E_m=25e3,
#                          reinforcement_lst=[reinf],
#                          n_BC=8)
    reinf1 = ContinuousFibers(r=3.5e-3,
                              tau=RV('weibull_min', loc=0.01, scale=.1, shape=2.),
                              V_f=0.005,
                              E_f=200e3,
                              xi=fibers_MC(m=7., sV0=0.005),
                              label='carbon',
                              n_int=200)

    ccb = RandomBondCB(E_m=25e3,
                                 reinforcement_lst=[reinf],
                                 Ll=1e5,
                                 Lr=1e5,
                                 w=0.392516512956)
    
    
    print ccb.sig_c(0.1)



    