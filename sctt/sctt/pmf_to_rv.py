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
    
    from etsproxy.util.home_directory import \
        get_home_directory
 
    import os.path
 
    home_dir = get_home_directory()
        
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
    data = interp(w_arr)
    
    cali = Calibration(m = 13.,
                       data=data,
                       w_arr=w_arr,
                       tau_arr=np.logspace(np.log10(1e-5), np.log10(1), 200))
    
    def residuum(arr):
        cali.sV0 = float(arr)
        sigma = cali.responses
        sigma[0] = 1e6*np.ones_like(cali.tau_arr)
        data[0] = 1e6
        residual = nnls(sigma, data)[1]
        return residual
    
    sV0 = brute(residuum, ((0.0001, 0.01),), Ns=20)

    sigma = cali.responses

    sigma[0] = 1e5*np.ones_like(cali.tau_arr)
    data[0] = 1e5
    
    x, y = nnls(sigma, data)
    
    pr = PMFtoRV(sample_points=cali.tau_arr,
                 weights=x)
    
    x_array = np.linspace(0, 0.2, 10000)
    pdf = pr.pdf(x_array)
    
    plt.plot(x_array, pdf)
    plt.show()
    


    
    
    
        
    
    