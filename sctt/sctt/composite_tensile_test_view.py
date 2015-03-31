from traits.api import \
    HasStrictTraits, Instance, Button, Event, Property, cached_property
from composite_tensile_test import CompositeTensileTest
from matplotlib.figure import Figure
from etsproxy.traits.ui.api import \
    View, Item, Group, HSplit, VGroup, HGroup, RangeEditor, InstanceEditor
from util.traits.editors.mpl_figure_editor import MPLFigureEditor
from random_fields.simple_random_field import SimpleRandomField
from crack_bridge_models.constant_bond_cb import ConstantBondCB
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.hom_CB_elastic_mtrx \
    import CompositeCrackBridge
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement \
    import ContinuousFibers
from spirrid.rv import RV
from util.traits.either_type import EitherType
from stats.pdistrib.weibull_fibers_composite_distr import \
    WeibullFibers, fibers_MC


class CompositeTensileTestView(HasStrictTraits):

    cb = EitherType(klasses=[ConstantBondCB, \
                             StochasticCB])
    # cb = Instance(StochasticCB)

    ctt = Instance(CompositeTensileTest)

    figure = Instance(Figure)
    def _figure_default(self):
        figure = Figure()
        return figure

    data_changed = Event

    data = Property(depends_on='cb.+m_para')
    @cached_property
    def _get_data(self):
        return self.ctt.get_cracking_history()

    eps_c_i = Property(depends_on='cb.+m_para')
    @cached_property
    def _get_eps_c_i(self):
        return self.ctt.get_eps_c_i(self.data[0], self.data[1], self.data[2])


    plot = Button
    def _plot_fired(self):
        axes1 = self.figure.add_subplot(111)
        axes1.plot(self.eps_c_i, self.data[0])
        axes1.plot([0.0, self.ctt.cb.eps_fu], [0.0, self.ctt.cb.sig_cu])
        self.data_changed = True

    clear = Button
    def _clear_fired(self):
        self.figure.clear()
        self.data_changed = True

    view = View(HSplit(Item('cb', style='custom', show_label=False),
                       Group(HGroup(Item('plot', show_label=False),
                                   Item('clear', show_label=False)),
                             Item('figure', editor=MPLFigureEditor(),
                                  dock='vertical', show_label=False))),
                resizable=True,
                height=600, width=800)

if __name__ == '__main__':

    reinf = ContinuousFibers(r=0.0035,
                          tau=RV('weibull_min', loc=0.0, shape=1., scale=1.),
                          V_f=0.01,
                          E_f=180e3,
                          xi=fibers_MC(m=2.0, sV0=0.003),
                          label='carbon',
                          n_int=200)

    cb1 = CompositeCrackBridge(E_m=25e3,
                               reinforcement_lst=[reinf]
                                 )

    scb = StochasticCB(ccb=cb1)



    cbcb = ConstantBondCB()
    rf = SimpleRandomField()
    model = CompositeTensileTest(cb=cbcb,
                                 sig_mu_x=rf.field)
    cttv = CompositeTensileTestView(cb=scb,
                                  ctt=model)
    cttv.configure_traits()
