import numpy as np


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
