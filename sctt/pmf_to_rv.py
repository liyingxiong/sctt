from etsproxy.traits.api import Float, Property, cached_property, Int, \
     Instance, Array, HasTraits
import numpy as np
from scipy.interpolate import interp1d
from calibration import Calibration
from matplotlib import pyplot as plt
from scipy.optimize import brute, nnls


class PMFtoRV(HasTraits):
    
    sample_points = Array
    weights = Array
    
    cumulation = Property(depends_on='weights')
    @cached_property
    def _get_cumulation(self):
        return np.cumsum(np.hstack([0., self.weights]))
    
    cdf = Property(depends_on='sample_points, weights')
    @cached_property
    def _get_cdf(self):
        def f(x):
            if x > np.amax(self.sample_points):
                return 1.
            else:
                sample = np.hstack([0, self.sample_points])
                cdf = interp1d(sample, self.cumulation,\
                                bounds_error=False, fill_value=0.)
                return cdf(x)
        return np.vectorize(f)
    
    ppf = Property(depends_on='sample_points, weights')
    @cached_property
    def _get_ppf(self):
        return interp1d(self.cumulation, np.hstack([0, self.sample_points]))
    
    pdf = Property(depends_on='sample_points, weights')
    @cached_property
    def _get_pdf(self):
        sample = np.hstack([0, self.sample_points])
        dx = np.diff(sample)
        p_density = np.hstack([0, self.weights/dx])
        pdf = interp1d(sample, p_density, bounds_error=False, fill_value=0.)
        return pdf

if __name__ == '__main__':
    
    w_arr = np.linspace(0.0, np.sqrt(8.), 401) ** 2
    
#     from etsproxy.util.home_directory import \
#         get_home_directory
#  
    import os.path
 
    home_dir = 'D:\\Eclipse\\'
        
    path = [home_dir, 'git',
            'rostar',
            'scratch',
            'diss_figs',
            'CB1.txt']

    filepath = os.path.join(*path)
    
    data = np.zeros_like(w_arr)
    
    file1 = open(filepath, 'r')
    cb = np.loadtxt(file1, delimiter=';')
    test_xdata = -cb[:,2]/4. - cb[:,3]/4. - cb[:,4]/2.
    test_ydata = cb[:,1] / (11. * 0.445) * 1000
    interp = interp1d(test_xdata, test_ydata, bounds_error=False, fill_value=0.)
    exp_data = interp(w_arr)
    
    cali = Calibration(experi_data=exp_data,
                       w_arr=w_arr,
                       tau_arr=np.logspace(np.log10(1e-5), 0.5, 50),
                       bc=6.85,
                       sig_mu=3.4,
                       m = 9,
                       sV0=0.0045)
    
    
    pr = PMFtoRV(sample_points=cali.tau_arr,
                 weights=cali.tau_weights)
    
    x_array = np.linspace(0, 0.2, 1000)
    pdf = pr.pdf(x_array)
    
    plt.plot(x_array, pdf)
    print np.trapz(pdf, x_array)
    plt.show()
    


    
    
    
        
    
    