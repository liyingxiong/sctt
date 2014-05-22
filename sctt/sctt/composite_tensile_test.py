from etsproxy.traits.api import \
    HasStrictTraits, Instance, Int, Float, List, Array, Property, \
    cached_property
    
class CompositeTensileTest(HasStrictTraits):
    
#=============================================================================
# discretization of the specimen
#=============================================================================    
    n_x = Int(1001) #number of material points
    L = Float(1000.) #the specimen length - mm
    x = Property(depends_on='n_x, L') #coordinates of the material points
    @cached_property
    def _get_x(self):
        return np.linspace(0, self.L, self.n_x)
