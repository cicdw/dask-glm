from .algos import admm, gradient_descent, newton


class LogisticModel(object):

    def hessian(self):
        pass


    def gradient(self):
        pass


    def loglike(self):
        pass


    def fit(self, tol=1e-8, warm_start=None, random_state=None, max_iter=250):
        pass


    def _initialize(self):
        pass


    def _check_inputs(self):
        pass


    def _set_intercept(self):
        pass


    def __init__(self, X, y, reg=None, lamduh=0,
                 fit_intercept=True)
        pass
