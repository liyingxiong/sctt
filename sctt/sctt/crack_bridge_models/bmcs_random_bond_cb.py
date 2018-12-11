'''
Created on Nov 27, 2018

@author: rch
'''

from scipy.interpolate import interp1d
from view.plot2d import Viz2D, Vis2D
from view.ui import BMCSLeafNode
from view.window import BMCSModel, BMCSWindow, TLine

import numpy as np
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement \
    import ContinuousFibers
from .random_bond_cb import RandomBondCB
from spirrid.rv import RV
from stats.pdistrib.weibull_fibers_composite_distr import \
    WeibullFibers, fibers_MC
import traits.api as tr
import traitsui.api as tu


class Viz2DCBFieldVar(Viz2D):
    '''Plot adaptor for the pull-out simulator.
    '''
    label = 'sig-x'

    def plot(self, ax, vot, *args, **kw):
        vis2d = self.vis2d
        z_arr = np.linspace(-self.Ll, self.Lr, 30)

        load = vot
        sig_m = vis2d.get_sig_m_z(z_arr, self.Ll, self.Lr, load)
        ax.plot(z_arr, sig_m, label='load=' + str(load))

    Ll = tr.Float(0.1)
    Lr = tr.Float(0.1)

    traits_view = tu.View(tu.Item('Ll', full_size=True, resizable=True),
                          tu.Item('Lr', full_size=True))


class Viz2DSigCMax(Viz2D):
    '''Plot adaptor for the pull-out simulator.
    '''
    label = 'sig-x'

    def plot(self, ax, vot, *args, **kw):
        vis2d = self.vis2d
        z_arr = np.linspace(0, 150, 300)


class BMCSRandomBondCB(RandomBondCB, BMCSModel, Vis2D):

    node_name = 'Random bond crack bridge'

    tree_node_list = tr.List([])

    def _tree_node_list_default(self):

        return [
        ]

    def _init_state_arrays(self):
        pass

    def init(self):
        if self._paused:
            self._paused = False
        if self._restart:
            self.tline.val = self.tline.min
            self.tline.max = 1
            self._restart = False
            self._init_state_arrays()

    def eval(self):
        self.interps

    def paused(self):
        self._paused = True

    def stop(self):
        self._sv_hist_reset()
        self._restart = True
        self.loading_scenario.reset()

    _paused = tr.Bool(False)
    _restart = tr.Bool(True)

    tline = tr.Instance(TLine)

    def _tline_default(self):
        # assign the parameters for solver and loading_scenario
        t_max = 1.0  # self.loading_scenario.t_max
        d_t = 0.1  # self.loading_scenario.d_t
        return TLine(min=0.0, step=d_t, max=t_max,
                     time_change_notifier=self.time_changed,
                     time_range_change_notifier=self.time_range_changed
                     )

    def get_time_idx_arr(self, vot):
        '''Get the index corresponding to visual time
        '''
        #x = np.array(self.sig_c_lst, dtype=np.float_)
        x = np.array(self.cc_lst, dtype=np.float_)
        idx = np.array(np.arange(len(x)), dtype=np.float_)
        t_idx = np.interp(vot, x, idx)
        return np.array(t_idx, np.int_)

    def get_time_idx(self, vot):
        return int(self.get_time_idx_arr(vot))


if __name__ == '__main__':

    reinf = ContinuousFibers(r=3.5e-3,
                             tau=RV(
                                 'gamma', loc=0.,
                                 scale=2.3273933214348754,
                                 shape=0.04793148098051675),
                             V_f=0.010,
                             E_f=180e3,
                             xi=fibers_MC(m=6, sV0=0.0095),
                             label='carbon',
                             n_int=500)

    rb_cb = BMCSRandomBondCB(E_m=25e3,
                             reinforcement_lst=[reinf],
                             Ll=100.,
                             Lr=150.,
                             L_max=300,
                             n_BC=6)

    viz2d_sig_eps = Viz2DCBFieldVar(name='field variable',
                                    vis2d=rb_cb)

    w = BMCSWindow(model=rb_cb)

    w.viz_sheet.viz2d_list.append(viz2d_sig_eps)
    w.viz_sheet.n_cols = 1
    w.viz_sheet.monitor_chunk_size = 1

    w.run()
    w.offline = False
    w.configure_traits()
