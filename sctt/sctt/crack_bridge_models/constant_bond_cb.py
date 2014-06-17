from etsproxy.traits.api import \
    HasStrictTraits, Int, Float, Property, cached_property
import numpy as np
from scipy.interpolate import interp2d
from etsproxy.traits.ui.api import \
    View, Item, Group, HSplit, VGroup, HGroup, RangeEditor, InstanceEditor


class ConstantBondCB(HasStrictTraits):

#=============================================================================
# material parameters
#=============================================================================    
    T = Float(12., m_para=True)  # Bond intensity [MPa/mm]
    E_m = Float(25e+3, m_para=True)  # [MPa]
    E_f = Float(180e+3, m_para=True)  # [MPa]
    v_f = Float(0.01, m_para=True)  # [-]
    E_c = Property(depends_on = 'E_m, E_f, v_f') #composite modulus
    @cached_property
    def _get_E_c(self):
        return self.E_m * (1 - self.v_f) + self.E_f * self.v_f
    
    sig_fu = Float(1.8e+3, m_para=True)  # reinforcement strength [MPa]
    sig_cu = Property(depends_on = 'sig_fu, v_f') # ultimate composite stress
    @cached_property
    def _get_sig_cu(self):
        return self.sig_fu * self.v_f
    eps_fu = Property(depends_on = 'sig_fu, E_f') # ultimate reinforcement strain
    @cached_property
    def _get_eps_fu(self):
        return self.sig_fu / self.E_f

#=============================================================================
# interpolation parameters
#=============================================================================    
    cbL = Float(200., i_para=True) #length [mm]
    n_z = Int(1000, i_para=True) #number of material points
    #coordinates of the material points
    cb_z = Property(depends_on='cbL, n_z')
    @cached_property
    def _get_cb_z(self):
        return np.linspace(0, self.cbL, self.n_z)
    
    n_sig_c = Int(100, i_para=True) 
    #composite stress levels
    cb_sig_c = Property(depends_on='n_sig_c, sig_fu, v_f')
    @cached_property
    def _get_cb_sig_c(self):
        return np.linspace(0, self.sig_cu, self.n_sig_c) 
    
#=============================================================================
# matrix stress and reinforcement strain profiles
#=============================================================================    
    cb_sig_m = Property(depends_on='+m_para, +i_para')
    @cached_property
    def _get_cb_sig_m(self):
        cb_sig_m = self.T * self.v_f / (1 - self.v_f) * self.cb_z
        return cb_sig_m.clip(max= \
                self.cb_sig_c[:, np.newaxis] / self.E_c * self.E_m)

    cb_eps_f = Property(depends_on='+m_para, +i_para')
    @cached_property
    def _get_cb_eps_f(self):
        cb_eps_f = self.cb_sig_c[:, np.newaxis] / (self.E_f * self.v_f) - \
                    self.T * self.cb_z / self.E_f
        return cb_eps_f.clip(min= self.cb_sig_c[:, np.newaxis] / self.E_c)

#=============================================================================
# interpolation function
#=============================================================================    
    # function for evaluating specimen matrix stress
    interp_sig_m_z = Property(depends_on='+m_para, +i_para')
    @cached_property
    def _get_interp_sig_m_z(self):
        return interp2d(self.cb_z, self.cb_sig_c, self.cb_sig_m, kind='linear')
    
    def get_sig_m_z(self, z_arr, ll, lr, load):
        return self.interp_sig_m_z(z_arr, load)

    # function for evaluating specimen reinforcement strain
    interp_eps_f_z = Property(depends_on='+m_para, +i_para')
    @cached_property
    def _get_interp_eps_f_z(self):
        return interp2d(self.cb_z, self.cb_sig_c, self.cb_eps_f, kind='linear')
        
    def get_eps_f_z(self, z_arr, ll, lr, load):
        return  self.interp_eps_f_z(z_arr, load)
    
#=============================================================================
# UI
#=============================================================================
    view = View(Group(Item('T', label='Bond intensity'),
                      Item('E_m', label='Matrix modulus'),
                      Item('E_f', label='Reinforcement modulus'),
                      Item('v_f', label='Reinforcement ratio'),
                      Item('cbL', label='Crack bridge length'),
                      Item('n_z', label='Number of material points'),
                      label='Crack Bridge Model', show_border=True))
    

if __name__ == '__main__':
    
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    ll, lr = 0, 0
        
    cbcb = ConstantBondCB(n_z = 1000)
    cbcb1 = ConstantBondCB(n_z = 1000)

#     cbcb.configure_traits()
    x = np.linspace(0, 1000, 10001)
    sig = cbcb.get_sig_m_z(x, ll, lr, cbcb.sig_cu/4)
    eps = cbcb.get_eps_f_z(x, ll, lr, cbcb.sig_cu/4)
    
    sig1 = cbcb1.get_sig_m_z(x, ll, lr, cbcb.sig_cu/4)
    eps1 = cbcb1.get_eps_f_z(x, ll, lr, cbcb.sig_cu/4)
    
    print sig == sig1
    
    plt.plot(x, sig - sig1)
#     plt.plot(x, sig1)
    
    plt.show()
#     fig = plt.figure(figsize=(10,10))
#     ax1 = fig.add_subplot(221, projection='3d')
#     X, Y = np.meshgrid(cbcb.cb_z, cbcb.cb_sig_c)
#     ax1.plot_wireframe(X, Y, cbcb.cb_sig_m, rstride=10, cstride=20)
#     ax2 = fig.add_subplot(222)
#     ax2.plot(x, sig)
#     ax2.plot(x, sig1)
#     ax3 = fig.add_subplot(223, projection='3d')
#     ax3.plot_wireframe(X, Y, cbcb.cb_eps_f, rstride=10, cstride=20)
#     ax4 = fig.add_subplot(224)
#     ax4.plot(x, eps)
#     plt.show()



    
    