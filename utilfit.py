import matplotlib.pyplot as mp
import numpy as np
from scipy import optimize

def fitGaussian(histogram, key, selection, normalized = False):
    
    hg = histogram[{key:selection}]
    values, variances = ([0] * (hg.size - 2) for blank in range(2))
    for i, element in enumerate(hg):
        if i != 0 and i != hg.size - 1:
            values[i], variances[i] = element.value, element.variance
    centres = np.linspace(-0.99, 0.99, 100)
            
    norm = np.sum(values) / (hg.size - 2)
    if normalized: values = np.multiply(norm, values)
    
    def gaussian(x, *p):
        N, mu, sigma = p
        return (N / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu)**2 / (2 * sigma**2))
    
    parameters = optimize.curve_fit(
        gaussian,
        centres, values,
        p0 = [1, 0, 1])[0]
    fit = gaussian(centres, *parameters)
    
    mp.title('Fit ' + key + ' ' + str(selection))
    mp.plot(centres, fit, color = 'b', label = 'Gaussian Fit')
    mp.stairs(values, np.linspace(-1, 1, 101), color = 'k', label = 'Data')
    mp.xlabel(histogram.name)
    mp.legend()
    mp.show()
    
    return parameters