

from matplotlib import pyplot as plt
from scipy.integrate import cumtrapz
from scipy.interpolate import interp2d, griddata
from scipy.optimize import root, fminbound, brute
from traits.api import HasTraits, Array, List, Float, Int, \
    Property, cached_property

import numpy as np
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement import \
    ContinuousFibers
from sctt.reinforcements.fiber_bundle import FiberBundle
from spirrid.rv import RV
from stats.pdistrib.weibull_fibers_composite_distr import WeibullFibers
import time as t
import traitsui.api as tu
from util.traits.either_type import EitherType


class RandomBondCB(HasTraits):

    #=========================================================================
    # Material Parameters
    #=========================================================================
    reinforcement_lst = List(
        EitherType(klasses=[FiberBundle, ContinuousFibers]))

    E_m = Float(25e3, auto_set=False, enter_set=True, MAT=True,
                desc='Elastic modulus of matrix')
    w = Float  # the crack width
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

    #=========================================================================
    # sort the integral points according to bond intensity in descending order
    #=========================================================================
    sorted_theta = Property(depends_on='reinforcement_lst+')

    @cached_property
    def _get_sorted_theta(self):
        depsf_arr = np.array([])  # bond intensity, 2*tau/r
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
            stat_weights_arr = np.hstack(
                (stat_weights_arr, reinf.stat_weights))
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
        max_depsf = [np.max(reinf.depsf_arr)
                     for reinf in self.reinforcement_lst]
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
# TODO: does not work for reinforcement types with the same xi: line 135
        methods = []
        masks = []
        for reinf in self.reinforcement_lst:
            masks.append(self.sorted_xi == reinf.xi)
            if isinstance(reinf.xi, float):
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

    #=========================================================================
    # evaluate maximum fiber strain and matrix strain profile
    #=========================================================================
    def dem_depsf_vect(self, damage):
        '''evaluates the deps_m given deps_f at that point and the damage array
        Eq.(4.17), return the array containing the matrix strain  derivative at
        each point z_i
        '''
        Kf_intact = self.Kf * (1. - damage)
        Kf_intact_bonded = np.hstack((0.0, np.cumsum((Kf_intact))))[:-1]
        Kf_broken = np.sum(self.Kf - Kf_intact)
        Kf_add = Kf_intact_bonded + Kf_broken
        Km = (1. - self.V_f_tot) * self.E_m
        E_mtrx = Km + Kf_add  # Eq.(4.15)
        mu_T = np.cumsum((self.sorted_depsf * Kf_intact)[::-1])[::-1]
        return mu_T / E_mtrx

    def F(self, dems, amin):
        '''Eq.(D.21)'''
        f = 1. / (self.sorted_depsf + dems)
        F = np.hstack((0., cumtrapz(f, -self.sorted_depsf)))
        return F

#     def F(self, dems, amin):
#         '''Auxiliary function (see Part II, appendix B)
#         '''
#         F = np.zeros_like(self.sorted_depsf)
#         for i, mask in enumerate(self.sorted_masks):
#             depsfi = self.sorted_depsf[mask]
#             demsi = dems[mask]
#             fi = 1. / (depsfi + demsi)
#             F[mask] = np.hstack((np.array([0.0]), cumtrapz(fi, -depsfi)))
#             if i == 0:
#                 C = 0.0
#             else:
#                 depsf0 = self.sorted_depsf[self.sorted_masks[i - 1]]
#                 depsf1 = depsfi[0]
#                 idx = np.sum(depsf0 > depsf1) - 1
#                 depsf2 = depsf0[idx]
#                 a1 = np.exp(F[self.sorted_masks[i - 1]][idx] / 2. + np.log(amin))
#                 p = depsf2 - depsf1
#                 q = depsf1 + demsi[0]
#                 amin_i = np.sqrt(a1 ** 2 + p / q * a1 ** 2)
#                 C = np.log(amin_i / amin)
#             F[mask] += 2 * C
#         return F

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
        amin = np.sqrt(
            self.w / (np.abs(init_dem) + np.abs(self.sorted_depsf[0])))
        # integrated f(depsf) - see article
        F = self.F(dems, amin)
        # a1 is a(depsf) for double sided pullout
        a1 = amin * np.exp(F / 2.)
#         aX = np.exp((-np.log(np.abs(self.sorted_depsf) + dems) + np.log(self.w)) / 2.)
        if Lmin < a1[0] and Lmax < a1[0]:
            # all fibers debonded up to Lmin and Lmax
            a, em, epsf0 = self.clamped(Lmin, Lmax, init_dem)

        elif Lmin < a1[0] and Lmax >= a1[0]:
            # all fibers debonded up to Lmin but not up to Lmax
            a2 = np.sqrt(2 * Lmin ** 2 + np.exp(F) * 2 *
                         self.w / (self.sorted_depsf[0] + init_dem)) - Lmin
            if Lmax < a2[0]:
                a, em, epsf0 = self.clamped(Lmin, Lmax, init_dem)
            else:
                if Lmax <= a2[-1]:
                    idx = np.sum(a2 < Lmax) - 1
                    a = np.hstack((-Lmin, 0.0, a2[:idx + 1], Lmax))
                    em2 = np.cumsum(np.diff(np.hstack((0.0, a2))) * dems)
                    em = np.hstack(
                        (init_dem * Lmin, 0.0, em2[:idx + 1], em2[idx] + (Lmax - a2[idx]) * dems[idx]))
                    um = np.trapz(em, a)
                    epsf01 = em2[:idx + 1] + a2[:idx + 1] * \
                        self.sorted_depsf[:idx + 1]
                    epsf02 = (
                        self.w + um + self.sorted_depsf[idx + 1:] / 2. * (Lmin ** 2 + Lmax ** 2)) / (Lmin + Lmax)
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
            # first debonded length amin for one sided pull out
#             depsfLmin = self.sorted_depsf[idx1]
#             p = (depsfLmin + dems[idx1])
#             a_short = np.hstack((a1[:idx1], Lmin))
#             em_short = np.cumsum(np.diff(np.hstack((0.0, a_short))) * dems[:idx1 + 1])
#             emLmin = em_short[-1]
#             umLmin = np.trapz(np.hstack((0.0, em_short)), np.hstack((0.0, a_short)))
#             amin = -Lmin + np.sqrt(4 * Lmin ** 2 * p ** 2 - 4 * p * emLmin * Lmin + 4 * p * umLmin - 2 * p * Lmin ** 2 * depsfLmin + 2 * p * self.w) / p
#             C = np.log(amin ** 2 + 2 * amin * Lmin - Lmin ** 2)
#             a2 = (np.sqrt(2 * Lmin ** 2 + np.exp(F + C - F[idx1])) - Lmin)[idx1:]
            a2 = np.sqrt(
                2 * Lmin ** 2 + np.exp(F[idx1:]) * 2 * self.w / (self.sorted_depsf[0] + init_dem)) - Lmin
            # matrix strain profiles - shorter side
            a_short = np.hstack((-Lmin, -a1[:idx1][::-1], 0.0))
            dems_short = np.hstack((dems[:idx1], dems[idx1]))
            em_short = np.hstack(
                (0.0, np.cumsum(np.diff(-a_short[::-1]) * dems_short)))[::-1]
            if a2[-1] > Lmax:
                idx2 = np.sum(a2 <= Lmax)
                # matrix strain profiles - longer side
                a_long = np.hstack((a1[:idx1], a2[:idx2]))
                em_long = np.cumsum(
                    np.diff(np.hstack((0.0, a_long))) * dems[:idx1 + idx2])
                a = np.hstack((a_short, a_long, Lmax))
                em = np.hstack(
                    (em_short, em_long, em_long[-1] + (Lmax - a_long[-1]) * dems[idx1 + idx2]))
                um = np.trapz(em, a)
                epsf01 = em_long + a_long * self.sorted_depsf[:idx1 + idx2]
                epsf02 = (
                    self.w + um + self.sorted_depsf[idx1 + idx2:] / 2. * (Lmin ** 2 + Lmax ** 2)) / (Lmin + Lmax)
                epsf0 = np.hstack((epsf01, epsf02))
            else:
                a_long = np.hstack((0.0, a1[:idx1], a2, Lmax))
                a = np.hstack((a_short, a_long[1:]))
                dems_long = dems
                em_long = np.hstack(
                    (np.cumsum(np.diff(a_long[:-1]) * dems_long)))
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
            a_short = np.hstack(
                (a_short, Lmin * np.ones(len(self.sorted_depsf) - len(a_short))))
        a_long = a[a > 0.0][:-1]
        if len(a_long) < len(self.sorted_depsf):
            a_long = np.hstack(
                (a_long, Lmax * np.ones(len(self.sorted_depsf) - len(a_long))))
        return epsf0, a_short, a_long

    #=========================================================================
    # calculate the damage portion of the reinforcement
    #=========================================================================
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
            elif method.__doc__ == 'weibull_fibers_cdf_dry':
                Pf[masks[i]] += method(epsy[masks[i]],
                                       self.sorted_r[masks[i]],
                                       x_short[masks[i]] + x_long[masks[i]])
#                                        1000)
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
            residuum = self.vect_xi_cdf(
                epsf0, x_short=x_short, x_long=x_long) - iter_damage
            return residuum

    damage = Property(depends_on='w, Ll, Lr, reinforcement+')

    @cached_property
    def _get_damage(self):
        if self.w == 0.:
            return np.zeros_like(self.sorted_depsf)
        else:
            damage = root(self.damage_residuum, np.ones_like(self.sorted_depsf) * 0.2,
                          method='excitingmixing', options={'maxiter': 200})
            if np.any(damage.x < 0.0) or np.any(damage.x > 1 + 1e-3):
                raise ValueError
            return damage.x

    #=========================================================================
    # composite stress at crack position as a function of crack width w
    #=========================================================================
    def sig_c(self, w):
        self.w = float(w)
        self.damage
        sig_c = np.sum(self._epsf0_arr * self.sorted_stats_weights
                       * self.sorted_V_f * self.sorted_nu_r
                       * self.sorted_E_f * (1. - self.damage))
        return sig_c

    def minus_sig_c(self, w):
        self.w = float(w)
        self.damage
        sig_c = np.sum(self._epsf0_arr * self.sorted_stats_weights
                       * self.sorted_V_f * self.sorted_nu_r
                       * self.sorted_E_f * (1. - self.damage))
        return -sig_c
    #=========================================================================
    # maximize sig_c and return the corresponding crack width w_max for given Ll, Lr
    #=========================================================================

    def max_sig_c(self, Ll, Lr):  # need further improvement
        self.Ll = Ll
        self.Lr = Lr

        def minus_sig_c(w): return -self.sig_c(w)
        w_upper_bound = min(0.1 * (self.Ll + self.Lr), 20)
        # determine the bound for fminbound
        mid = brute(self.minus_sig_c, ((1e-3, w_upper_bound),), Ns=10,
                    finish=None)
#         print res
#
#         mid = res[0]

        results = fminbound(self.minus_sig_c, max(0, mid - 0.1 * w_upper_bound),
                            mid + 0.1 * w_upper_bound,
                            maxfun=50, full_output=1, disp=0)
        max_w = results[0]
        max_sig_c = -results[1]
        return max_sig_c, max_w

    #=========================================================================
    # interpolation parameters
    #=========================================================================
    # number of different boundary conditions in the interpolator
    n_BC = Int(8)
    L_max = Float  # maximum debonded length, equals to the specimen length

    n_Int = Property(depends_on='reinforcement_lst')

    @cached_property
    def _get_n_Int(self):
        n_Int = 0
        for reinf in self.reinforcement_lst:
            n_Int += reinf.n_int
        return n_Int

    BC_range = Property(depends_on='reinforcement_lst+, E_m')
    '''
    Discretization of the boundary condition
    '''
    @cached_property
    def _get_BC_range(self):
        self.w = self.max_sig_c(1e5, 1e5)[1]
        self.damage
        L_max = min(self._x_arr[-2], self.L_max)  # maximum debonding length
        BC_range = np.logspace(
            np.log10(0.02 * L_max), np.log10(L_max), self.n_BC)
        return BC_range

    interps = Property(denpends_on='reinforcement_lst+, Ll, Lr, n_BC, E_m')
    '''
    Prepare the interpolator for each boundary condition
    '''
    @cached_property
    def _get_interps(self):
        interps_sigm = []
        interps_epsf = []
        sig_max_lst = []
#         interps_damage = []
        t1 = t.clock()
        print('preparing the interpolators:')
        for j, L_r in enumerate(self.BC_range):
            for q, L_l in enumerate(self.BC_range):
                if L_l <= L_r:
                    sigma_max, w_max = self.max_sig_c(L_l, L_r)

                    # w is discretized in such a way that the corresponding sig_c
                    # are more evenly distributed
                    w_arr = np.linspace(1e-15, np.sqrt(w_max), 20) ** 2
                    # a uniform x coordinates for interpolation
                    x_arr_record = np.linspace(0, L_r, self.n_Int)

                    epsf_record = []
                    sigm_record = []
                    sigc_record = []
#                     damage_record = []

                    for w in w_arr:
                        self.w = w
                        self.damage

                        epsf_x = np.zeros_like(self._x_arr)
                        for i, depsf in enumerate(self.sorted_depsf):
                            epsf_x += np.maximum(self._epsf0_arr[i] -
                                                 depsf * np.abs(self._x_arr),
                                                 self._epsm_arr) * self.sorted_stats_weights[i]

                        sigm_x = self._epsm_arr * self.E_m

                        sigc = np.sum(self._epsf0_arr * self.sorted_stats_weights
                                      * self.sorted_V_f * self.sorted_nu_r
                                      * self.sorted_E_f * (1. - self.damage))

                        #======================================================
                        # reshape the strain and stress array to make the data
                        # on a regular grid
                        epsf_x = griddata(self._x_arr, epsf_x, x_arr_record)
                        sigm_x = griddata(self._x_arr, sigm_x, x_arr_record)
                        #======================================================

                        epsf_record.append(epsf_x)
                        sigm_record.append(sigm_x)
                        sigc_record.append(sigc)
#                         damage_record.append(np.average(self.damage, weights=self.sorted_stats_weights))

                    #==========================================================
                    # plot the stress or strain profile under given BC
#                     if L_l == self.BC_range[-1] and L_r == self.BC_range[-1]:
#                     plt.figure()
#                     X, Y = np.meshgrid(x_arr_record, sigc_record)
#                     fig = plt.figure()
#                     ax = fig.add_subplot(111, projection='3d')
#                     ax.plot_wireframe(X, Y, sigm_record, rstride=1, cstride=1)
                    #==========================================================

                    interp_epsf = interp2d(
                        x_arr_record, sigc_record, epsf_record)
                    interp_sigm = interp2d(
                        x_arr_record, sigc_record, sigm_record)
#                     interp_damage = interp1d(sigc_record, damage_record, bounds_error=False, fill_value=damage_record[-1])

#                     print ((j + 1) * j / 2 + q + 1) * 100 / \
#                         (self.n_BC * (self.n_BC + 1) / 2), '%'

                    interps_epsf.append(interp_epsf)
                    interps_sigm.append(interp_sigm)
                    sig_max_lst.append(sigma_max)
#                     interps_damage.append(interp_damage)

        print(('time consumed:', t.clock() - t1))
        return interps_epsf, interps_sigm, np.array(sig_max_lst)
#
    #=========================================================================
    # functions for evaluation of stress and strain profile
    #=========================================================================

    def get_index(self, Ll, Lr):
        # find the index of the interpolator corresponding to the BC
        l, r = np.sort([Ll, Lr])
        i = min(np.sum(self.BC_range - l < 0), self.n_BC - 1)
        j = min(np.sum(self.BC_range - r < 0), self.n_BC - 1)
        ij = int((j + 1) * j / 2 + i)
        return ij

    # function for evaluating specimen reinforcement strain
    def get_eps_f_z(self, z_arr, Ll_arr, Lr_arr, load):
        def get_eps_f_i(self, z, Ll, Lr, load):
            ind = self.get_index(Ll, Lr)
            f = self.interps[0][ind]
            return f(z, load)
        v = np.vectorize(get_eps_f_i)
        return v(self, z_arr, Ll_arr, Lr_arr, load)

    # function for evaluating specimen matrix stress
    def get_sig_m_z(self, z_arr, Ll_arr, Lr_arr, load):
        def get_sig_m_i(self, z, Ll, Lr, load):
            ind = self.get_index(Ll, Lr)
            print('indexes', ind)
            f = self.interps[1][ind]
            return f(z, load)
        v = np.vectorize(get_sig_m_i)
        return v(self, z_arr, Ll_arr, Lr_arr, load)

    traits_view = tu.View(
        tu.Item('E_m', full_size=True, resizable=True),
        tu.Item('Lr', style='readonly',
                full_size=True),
        tu.Item('Ll', style='readonly')
    )

    tree_view = traits_view


if __name__ == '__main__':

    #     reinf = FiberBundle(r=0.0035,
    #                       tau=np.array([ 0.48419826,  0.71887589,  0.86655175,  0.98916772,  1.10160428, 1.21166999,  1.32595141,  1.45322794,  1.61204695,  1.89332916]),
    #                       tau_weights = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    #                       V_f=0.1,
    #                       E_f=240e3,
    #                       xi=0.035)
    from stats.pdistrib.weibull_fibers_composite_distr import \
        fibers_MC

    reinf = ContinuousFibers(r=3.5e-3,
                             tau=RV(
                                 'gamma', loc=0., scale=2.3273933214348754, shape=0.04793148098051675),
                             V_f=0.010,
                             E_f=180e3,
                             xi=fibers_MC(m=6, sV0=0.0095),
                             label='carbon',
                             n_int=500)

    ccb = RandomBondCB(E_m=25e3,
                       reinforcement_lst=[reinf],
                       Ll=150.,
                       Lr=150.,
                       L_max=400,
                       n_BC=7)

    z_arr = np.linspace(0, 150, 300)
    Ll_arr = 150. * np.ones_like(z_arr)
    Lr_arr = 5. * np.ones_like(z_arr)
    for load in np.linspace(1.5, 4.5, 3):
        sig_m = ccb.get_sig_m_z(z_arr, Ll_arr, Lr_arr, load)
        plt.plot(z_arr, sig_m, label='load=' + str(load))
    plt.xlim((0, 120))
    plt.legend(loc='best', ncol=2)
    plt.show()
#
#     w_arr = np.linspace(1e-15, 0.6, 400)
#     sig_w_arr = []
#     damage_arr = []
#     for w in w_arr:
#         sig_w_arr.append(ccb.sig_c(w))
#         ccb.w = w
#         ccb.damage
#         damage_arr.append(np.average(ccb.damage, weights=ccb.sorted_stats_weights))
#
#
#     plt.rc('text', usetex=True)
#     plt.rc('font', family='serif')
#
#
#     fig, ax1 = plt.subplots()
#     ax1.plot(w_arr, sig_w_arr, 'k')
#     ax1.set_ylabel('stress [MPa]')
#     ax1.set_xlabel('crack opening [mm]')
#
#
#     ax2 = ax1.twinx()
#     ax2.plot(w_arr, damage_arr, 'k--')
#     ax2.set_ylabel('breaking probability')
#     plt.show()


#     reinf1 = FiberBundle(r=0.0035,
#                   tau=cali.tau_arr,
#                   tau_weights = cali.tau_weights,
#                   V_f=0.01,
#                   E_f=200e3,
#                   xi=fibers_MC(m=cali.m, sV0=cali.sV0))

#=========================================================================
#     sV0_arr = []
#     m_arr = [6., 7., 8., 9., 10., 11.]
#     for m in m_arr:
#
#         def cbstrength(sV0):
#             reinf = ContinuousFibers(r=3.5e-3,
#                                       tau=RV('gamma', loc=0.0, scale=1.49376289, shape=0.06158335),
#                                       V_f=0.010,
#                                       E_f=180e3,
#                                       xi=fibers_MC(m=m, sV0=sV0),
#                                       label='carbon',
#                                       n_int=500)
#
#             ccb = RandomBondCB(E_m=25e3,
#                                reinforcement_lst=[reinf],
#                                Ll=7.,
#                                Lr=400.,
#                                L_max = 400,
#                                n_BC=20)
#             return ccb.max_sig_c(7., 375.)[0]-12.01
#
#         sV0_arr.append(brentq(cbstrength,1e-15, 0.015))
#
#
#     strength = []
#     for i, m in enumerate(m_arr):
#         s = sV0_arr[i]
#         reinf = ContinuousFibers(r=3.5e-3,
#                           tau=RV('gamma', loc=0.0, scale=1.49376289, shape=0.06158335),
#                           V_f=0.015,
#                           E_f=180e3,
#                           xi=fibers_MC(m=7, sV0=0.0095),
#                           label='carbon',
#                           n_int=500)
#         ccb = RandomBondCB(E_m=25e3,
#                    reinforcement_lst=[reinf],
#                    Ll=7.,
#                    Lr=400.,
#                    L_max = 400,
#                    n_BC=20)
#
#         strength.append(ccb.max_sig_c(5., 375.)[0])
#
#     print strength
#     plt.plot(m_arr, strength)
#
#     plt.ylim((0, 25.))
#     plt.show()
#=========================================================================

#     plt.plot([6., 7., 8., 9., 10., 11.], sV0_arr)
#     plt.show()

#     strength1=[]
#     Lr_arr_1 = np.linspace(7, 400, 20)
#     for Lr in Lr_arr_1:
#         strength1.append(ccb.max_sig_c(7., Lr)[0])
#
#
#     reinf.V_f=0.0015
#     strength1_5=[]
#     Lr_arr_1_5 = np.linspace(5, 400, 20)
#     for Lr in Lr_arr_1_5:
#         strength1_5.append(ccb.max_sig_c(5., Lr)[0]*1.5)


#     plt.rc('text', usetex=True)
#     plt.rc('font', family='serif')
#     plt.plot(Lr_arr_1, strength1, 'black', marker='o', label=r'$v_{f}=1\%$, fiber-MC')
#     plt.plot(Lr_arr_1_5, strength1_5, 'black', marker='.', label=r'$v_{f}=1.5\%$, fiber-MC')
#     plt.xlabel('$L_{\downarrow}$ [mm]')
#     plt.ylabel('Strength [Mpa]')
#     plt.tick_params(labelsize=14)
#     plt.legend()
#     plt.show()

#     print ccb.interps[2]

#     print ccb.max_sig_c(8., 400.)

#
# w_arr = np.linspace(0, 3, 100)
#
#     sigma = []
# #
# print ccb.Ll, ccb.Lr
# #
# #
#     for w in w_arr:
#         sig_c = ccb.sig_c(w)
#         sigma.append(sig_c)
# #
# print np.amax(sigma)
# #
#     plt.subplot(111)
#     plt.plot(w_arr, sigma)
#     plt.plot(w_arr, exp_data*reinf.V_f)
#     plt.show()

#     for j, w in enumerate(w_arr):
#         ccb.w = w
#         ccb.damage
#         plt.figure(num=None, figsize=(12, 6))
#         plt.clf()
#         plt.subplot(121)
#         plt.plot(ccb._x_arr, ccb.E_m*ccb._epsm_arr)
#         plt.ylim((0,5))
#         plt.text(0, 0, ccb.sig_c(w))
#
#         plt.subplot(122)
#         plt.plot(np.zeros_like(ccb._epsf0_arr), ccb._epsf0_arr, 'ro', label='maximum')
#         for i, depsf in enumerate(ccb.sorted_depsf):
#             epsf_x = np.maximum(ccb._epsf0_arr[i] - depsf * np.abs(ccb._x_arr), ccb._epsm_arr)
# print np.trapz(epsf_x - ccb._epsm_arr, ccb._x_arr)
# if i == 0:
# plt.plot(ccb._x_arr, epsf_x, color='blue', label='fibers')
# else:
#             plt.plot(ccb._x_arr, epsf_x, color='black', alpha=1-ccb.damage[i])
#         plt.plot(ccb._x_arr, ccb._epsm_arr, lw=2, color='blue', label='matrix')
#         plt.legend(loc='best')
#         plt.ylabel('matrix and fiber strain [-]')
#         plt.ylabel('long. position [mm]')
#         plt.ylim((0., 0.05))
#         savepath = 'D:\cracking history\\1\\crack_opening'+str(j)+'.png'
#         plt.savefig(savepath)
# plt.close()


#     sig = []
#
#     for w in w_arr:
#         sig.append(ccb.sig_c(w))
# #
#     plt.plot(w_arr, sig)

# #
#     print ccb.interps[2]
#
#     print ccb.BC_range
#     print ccb.damage
#     plt.plot(ccb._x_arr, ccb.E_m*ccb._epsm_arr)

#     z_arr = np.linspace(0, 200, 200)
#     bc1 = 20*np.ones_like(z_arr)
#     sig_m = ccb.get_sig_m_z(z_arr, bc1, bc1, 6)
#     plt.plot(z_arr, sig_m)
#     print ccb._x_arr
#     print len(ccb._x_arr)
#     print ccb._epsf0_arr

# #
#     print ccb._epsf0_arr
#
#     plt.figure()
#     plt.plot(np.zeros_like(ccb._epsf0_arr), ccb._epsf0_arr, 'ro', label='maximum')
#     for i, depsf in enumerate(ccb.sorted_depsf):
#         epsf_x = np.maximum(ccb._epsf0_arr[i] - depsf * np.abs(ccb._x_arr), ccb._epsm_arr)
#         print np.trapz(epsf_x - ccb._epsm_arr, ccb._x_arr)
#         if i == 0:
#             plt.plot(ccb._x_arr, epsf_x, color='blue', label='fibers')
#         else:
#             plt.plot(ccb._x_arr, epsf_x, color='black')
#     plt.plot(ccb._x_arr, ccb._epsm_arr, lw=2, color='blue', label='matrix')
#     plt.legend(loc='best')
#     plt.ylabel('matrix and fiber strain [-]')
#     plt.ylabel('long. position [mm]')
#     plt.xlim((-500, 500))


#     print ccb.max_sig_c(ccb.Ll, ccb.Lr)
#     w_arr = np.linspace(1e-15, 1, 200)
#     sig = []
#     for w in w_arr:
#         sig.append(ccb.sig_c(w))
#     plt.figure()
#     plt.plot(w_arr, sig)
#     print ccb.sig_c(0.0421542820709)
#     w_max = ccb.max_sig_c(3.49735724, 3.49735724)[1]
#     print w_max
#     sig_w = []
#     em_w = []
#     for w in np.linspace(1e-15, w_max, 100):
#         sig_w.append(ccb.sig_c(w))
#         em_w.append(ccb._epsm_arr[-1])
#     plt.figure()
#     plt.plot(sig_w, em_w)
#     print ccb.sig_c(0.0421542820709)
#     plt.figure()
#     plt.plot(np.linspace(1e-15, w_max, 100), sig_w)
#     sig1=[]
#     for w in w_arr:
#         sig1.append(ccb.sig_c(w))
#     plt.figure()
#     plt.plot(w_arr, sig1)
#     print ccb.sig_c(0.0421542820709)
#     plt.show()
