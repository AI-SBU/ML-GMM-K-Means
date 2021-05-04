import numpy as np


class MVN:
    def __init__(self, mean, cov, n_samples, seed):
        self._mean = mean
        self._cov = cov
        self._n_samples = n_samples
        self._seed = seed

    '''
    generates MVN data based on the given parameter
    '''

    def gen_mvn_data(self):
        np.random.seed(self._seed)
        return np.random.multivariate_normal(mean=self._mean, cov=self._cov, size=self._n_samples)
