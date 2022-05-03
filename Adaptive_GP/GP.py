import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch as th
import torch.autograd as autograd
import torch.optim as optim
from torch.distributions import constraints, transform_to
import os
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist

smoke_test = ('CI' in os.environ)
pyro.set_rng_seed(0)

from datetime import datetime


class GP(object):
    def __init__(self, input_dim, function):
        """

        Parameters
        ----------
        input_dim :
        function :
        """

        self.input_dim = input_dim
        self.function = function
        self.gpr = None

        # TODO: data bounds

    def wrapper_fn(self, X):
        """

        Parameters
        ----------
        X :

        Returns
        -------

        """
        assert X.ndim == 2, "invalid input shape, it should be a 2d vector"
        assert X.shape[1] == self.input_dim, "invalid input dim"
        Y = self.function(X)
        return Y

    def ARD(self,model):
        """

        Parameters
        ----------
        model :

        Returns
        -------

        """

    def fit(self,X):
        """
        Fits the gaussian process at the points given points
        Parameters
        ----------
        X : numpy.ndarray
            Array of training points
        Returns
        -------

        """
        assert X.ndim == 2, "invalid input shape, it should be a 2d vector"
        assert X.shape[1] == self.input_dim, "invalid input dim"

        #X = np.reshape(X, (-1, 1))
        #x = th.from_numpy(X)
        # defining the gp object

        # TODO : Can use mattern52 kertnel also
        # kernel = gp.kernels.RBF(input_dim=feature_space_input_dim, variance=th.tensor(1.),
        #                         lengthscale=0.1*th.ones(feature_space_input_dim))

        # computing the exact function value
        y = self.wrapper_fn(X)
        # default kernel lengthscale = 1, variance =1
        self.gpr = gp.models.GPRegression(th.from_numpy(X), th.from_numpy(y), gp.kernels.Matern52(input_dim=self.input_dim), noise=th.tensor(0.1),
                                     jitter=1.0e-4)
        # Calling the ARD kernel
        # ARD(gpr,x)

        # learning
        optimizer = th.optim.Adam(self.gpr.parameters(), lr=0.001)
        loss = gp.util.train(self.gpr, optimizer)
        return loss

    # -- Set of fucntions related to adaptively fitting
    def adapt_fit(self,num_steps,eps=1e-6):
        """

        Parameters
        ----------
        num_steps :
        eps :

        Returns
        -------

        """
        # TODO: link it with the fit function. Also define the x_init points
        for i in range(num_steps):
            #x, fopt = self.adapt_maximizer.maximize(self.acquisition_obj.acquisition_curve, self.lower_bound, self.upper_bound)
            x_opt = self.multi_start_opt()
            if np.abs(fopt) > eps: # some stopping criterion
                y = self.wrapper_fn(x_opt)  # evaluate f at new point.
                X = th.cat([self.gpr.X, x_opt])  # incorporate new evaluation
                y = th.cat([self.gpr.y, y])
                self.gpr.set_data(X, y)
                # optimize the GP hyperparameters using Adam with lr=0.001
                optimizer = th.optim.Adam(self.gpr.parameters(), lr=0.001)
                gp.util.train(self.gpr, optimizer)
            else:
                print("Adaptivity stopped because of very low variance")
                break


    def acquisition_fn(self, X, kappa =2):
        """
        #TODO: Generalise it and add possibility to add other types also
        The lower confidence bound acq fn. Striking a balance between exploitation and exploration
        Parameters
        ----------
        X :
        kappa :

        Returns
        -------

        """
        #mu, variance = model(X, full_cov=False, noiseless=False)
        mu, cov = self.predict(X)
        sigma = cov.sqrt()
        return mu - kappa * sigma

    def optimise_acq_fn(self,x_init,lower_bound=0, upper_bound=1):
        """

        Parameters
        ----------
        x_init :
        lower_bound :
        upper_bound :

        Returns
        -------

        """
        # transform x to an unconstrained domain
        constraint = constraints.interval(lower_bound, upper_bound)
        unconstrained_x_init = transform_to(constraint).inv(x_init)
        unconstrained_x = unconstrained_x_init.clone().detach().requires_grad_(True)
        minimizer = optim.LBFGS([unconstrained_x], line_search_fn='strong_wolfe')

        def closure():
            minimizer.zero_grad()
            x = transform_to(constraint)(unconstrained_x)
            y = self.acquisition_fn(x)
            autograd.backward(unconstrained_x, autograd.grad(y, unconstrained_x))
            return y

        minimizer.step(closure)
        # after finding a candidate in the unconstrained domain,
        # convert it back to original domain.
        x = transform_to(constraint)(unconstrained_x)
        return x.detach()

    def multi_start_opt(self, lower_bound=0, upper_bound=1, num_candidates=5):
        """
        Grad based optimisation methods can get stuck in local minimum. The method adopted is as follows:
         - First, we seed our minimization algorithm (optimise_acq_fn) with 5 different values: i) one is chosen to be ,
          i.e. the candidate used in the previous step; and ii) four are chosen uniformly at random from the domain
          of the objective function.
        - We then run the minimization algorithm to approximate convergence for each seed value.
        - Finally, from the five candidate x's identified by the minimization algorithm, we select the one that
        minimizes the acquisition function.

        Parameters
        ----------
        lower_bound :
        upper_bound :
        num_candidates :

        Returns
        -------

        """

        candidates = []
        values = []

        x_init = self.gpr.X[-1:]
        for i in range(num_candidates):
            x = self.optimise_acq_fn(x_init, lower_bound, upper_bound)
            y = self.acquisition_fn(x)
            candidates.append(x)
            values.append(y)
            x_init = x.new_empty(1).uniform_(lower_bound, upper_bound)  # chosen uniformly at random from the domain
            # of the objective function, of the same size

        argmin = th.min(th.cat(values), dim=0)[1].item() # index for which y is min
        return candidates[argmin] # returns the best x

    # -- common functions -------------------------------------------------------------------------------------

    def predict(self, X):
        """

        Parameters
        ----------
        model :
        x :

        Returns
        -------

        """
        assert X.ndim == 2, "invalid input shape, it should be a 2d vector"
        assert X.shape[1] == self.input_dim, "invalid input dim"
        mean, cov = self.gpr(th.from_numpy(X), full_cov=False,noiseless=False)  # Remove or include observational noise
        # in prediction?

        return mean.detach().numpy(), cov.detach().numpy()

    def get_mse(self, xtest=None, ytest=None, num_points=100):
        """
        Get the mean square error for the given test points

        Parameters
        ----------
        xtest : numpy.ndarray (optional)
            Array of test points
        ytest : numpy.ndarray (optional)
            Array of value at the test points
        num_points : int (optional)
            Number of test points

        Return
        ------
        mse : double
            Mean square error
        """
        if xtest is None or ytest is None:
            Xtest = np.random.uniform(self.lower_bound, self.upper_bound, (num_points, self.input_dim))
            Ytest = self.function(Xtest)
        pred, _ = self.predict(xtest)
        loss = th.nn.MSELoss()
        output = loss(th.tensor(pred), th.tensor(Ytest))
        return output

    # -- plots ------------------------------
    def predict_plot(self,mean,cov,x_pred,x_fn_input):
        """

        Parameters
        ----------
        mean :
        cov :
        x_fn_input :
        x_pred:
        Returns
        -------

        """
        # TODO :  this wont work if fn is expensive
        y_fn = self.wrapper_fn(x_fn_input)
        with th.no_grad():
            plt.plot(x_fn_input,y_fn, 'r',label='Exact function')
            plt.plot(self.gpr.X.numpy(), self.gpr.y.numpy(), "kx", label='Observations' )
            sd = np.sqrt(cov)
            plt.plot(x_pred,mean,'c',label='GP')
            plt.fill_between(x_pred.flatten(), mean - 2 * sd, mean + 2 * sd,
                             color="C0", alpha=0.3)  # plot uncertainty intervals
            plt.legend()
            plt.show()

class diagnostics:
    @staticmethod
    def predict_plot():
        raise NotImplementedError
