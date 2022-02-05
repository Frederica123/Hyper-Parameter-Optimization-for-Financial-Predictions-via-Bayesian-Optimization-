class ACQ:
    def acq_fn(self, *args, **kwargs):
        raise NotImplemented
    def __str__(self):
        return self.__class__.__name__
    def __call__(self, *args, **kwargs):
        return self.acq_fn(*args, **kwargs)

class ACQ1(ACQ):
    def acq_fn(self, gp, x, lam = 0.4, **kwrags):
        """
        gp: sklearn.GPRegresssor
        lam: float, where the objective is: \mu(x) + \lambda \sigma(x)
        """
        y_pred, var = [t.flatten() for t in gp.predict(x)]
        sigma = np.sqrt(var).squeeze()
        
        return y_pred + lam*sigma


class PI(ACQ):
    def acq_fn(
        self, gp, x, mu=5., eps=0.01, **kwargs):
        """
        gp: sklearn.GPRegresssor
        """
        y_pred, var = [t.flatten() for t in gp.predict(x)]
        sigma = np.sqrt(var).squeeze()
        
        cdf = ndtr((y_pred - mu - eps)/sigma)   #Returns the area under the standard Gaussian probability density function,
        #integrated from minus infinity to x
        return cdf

class EI(ACQ):
    def acq_fn(self, gp, x, mu=5., eps=0.01, **kwargs):
        """
        gp: sklearn.GPRegresssor
        mu: max value of y among the selected train_pts
        """
        y_pred, var = [t.flatten() for t in gp.predict(x)]
        sigma = np.sqrt(var).squeeze()
        z = (y_pred - mu - eps)/sigma
        return (y_pred - mu - eps)*ndtr(z) + sigma*norm.pdf(z)

class Thompson(ACQ):
    def acq_fn(self, gp, x, mu=5., **kwargs):
        """
        gp: sklearn.GPRegresssor
        mu: max value of y among the selected train_pts
        x: domain in which we are optimizing
        """
        sampled_y = gp.posterior_samples_f(x, size=1)
        return sampled_y.flatten()