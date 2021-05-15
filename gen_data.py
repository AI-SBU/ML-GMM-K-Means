import numpy as np

'''
The class below contains a function which is used to generate samples
from multivariate normal distribution
Three parameters are needed to construct an object of this class
The parameters are : mean, covariance and number of samples, respectively
'''


class MVN:
    def __init__(self, mean, cov, n_samples):
        self._mean = mean
        self._cov = cov
        self._n_samples = n_samples

    '''
    generates MVN data based on the given parameter
    '''

    def gen_mvn_data(self):
        np.random.seed(42)
        return np.random.multivariate_normal(mean=self._mean, cov=self._cov, size=self._n_samples)
