from sklearn import linear_model
import numpy as np
import scipy.stats as stat


class LinearRegressionWrapper(linear_model.LinearRegression):
    def __init__(self, fit_intercept=True, copy_X=True, n_jobs=1):
        super().__init__(fit_intercept=fit_intercept,  n_jobs=n_jobs)
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.feature_names = []
        
    def fit(self, X, y, n_jobs=1):
        super().fit(X, y, n_jobs)
        sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
        X = X.astype('float64')
        se = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X.T, X))))])
        self.t = self.coef_ / se
        self.p = np.squeeze(2 * (1 - stat.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1])))
        return self
