from etsproxy.traits.api import \
    HasStrictTraits, Float, Int, Property, cached_property
import numpy as np

class SimpleRandomField(HasStrictTraits):
    
    n_x = Int(1001, input=True) # number of material points
    mean = Float(3., input=True)
    deviation = Float(0.3, input=True)
    
    
    f_arr = np.linspace(-5, 5, 51)
    filter_ = np.exp(-1.*(f_arr**2/10))/np.sum(np.exp(-1.*(f_arr**2/10)))
     
    field = Property(depends_on='+input')
    @cached_property
    def _get_field(self):
        return self.mean + self.deviation*np.convolve( \
                np.random.randn(self.n_x), self.filter_, mode='same')
    
if __name__ == '__main__':
    
    srf = SimpleRandomField()
    x = np.linspace(0, 100, srf.n_x)
    from matplotlib import pyplot as plt
    plt.plot(x, srf.field)
    plt.show()

    

    
    
    
    