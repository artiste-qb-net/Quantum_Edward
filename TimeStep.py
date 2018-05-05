import numpy as np


class TimeStep:
    """
    An iteration or time step 't' is each time the parameters lambda = 
    list1_conc0, list1_conc1 and the function of lambda being MAXIMIZED ( 
    ELBO) are changed. This class calculates the change in lambda, 
    delta lambda, for a single iteration at the current time t=cur_t. There 
    are various possible methods for calculating delta lambda. A nice 
    description of the various methods can be found in the Wikipedia article 
    (cited below) for "Stochastic Gradient Descent" (in our case, it's an 
    ascent) 

    We use here a positive eta (eta is a scalar factor multiplying delta 
    lambda) because we are trying to maximize ELBO. In conventional 
    Artificial Neural Net algorithms, one is minimizing cost, so one uses a 
    negative eta. 

    References
    ----------
    https://en.wikipedia.org/wiki/Stochastic_gradient_descent

    """

    def __init__(self, method, eta, len1):
        """
        Constructor

        Parameters
        ----------
        method : str
            A string that identifies the method of calculating delta lambda.
            For example, method = 'adam'
        eta : float
            positive scalar, delta lambda is proportional to it.
        len1 : int
            length of a list1

        Returns
        -------
        None

        """
        self.method = method
        self.eta = eta

        # These are used by successive calls to get_delta_conc()
        self.list1_cum_grad = [None]*len1
        self.list1_cum_sq_grad = [None]*len1


    def get_delta_conc(self, grad0, grad1, cur_t, k):
        """
        Change in lambda = concentrations conc0 and conc1. grad0 and grad1
        are the gradients of ELBO at time t=cur_t with respect to conc0 and
        conc1, respectively. ELBO is being maximized. 'k' is the layer being
        considered.

        Parameters
        ----------
        grad0 : np.array
        grad1 : np.array
        cur_t : int
        k : int

        Returns
        -------
        np.array

        """
        method = self.method
        grad = np.stack([grad0, grad1])

        if method == 'naive':
            return self.eta*grad
        elif method == 'naive_t':
            return (self.eta/(cur_t+1))*grad
        elif method == 'mag1_grad':
            # mag_grad = magnitude of gradient
            mag_grad = np.sqrt(np.square(grad0) +
                               np.square(grad0))
            return np.divide(self.eta*grad, mag_grad)
        elif method == 'ada_grad':
            assert cur_t is not None
            if cur_t == 0:
                self.list1_cum_sq_grad[k] = np.square(grad)
                # print("**************", self.list1_cum_sq_grad[k])
            else:
                # print('..', cur_t, self.list1_cum_sq_grad[k])
                self.list1_cum_sq_grad[k] += np.square(grad)
            return np.divide(self.eta * grad,
                             np.sqrt(self.list1_cum_sq_grad[k]) + 1e-6)
        elif method == 'adam':
            assert cur_t is not None
            if cur_t == 0:
                self.list1_cum_grad[k] = grad
                self.list1_cum_sq_grad[k] = np.square(grad)
            else:
                self.list1_cum_grad[k] += grad
                self.list1_cum_sq_grad[k] += np.square(grad)
            return np.divide(np.multiply(self.eta * grad,
                                         self.list1_cum_grad[k]),
                             np.sqrt(self.list1_cum_sq_grad[k]) + 1e-6)
        else:
            assert False, "unsupported time step method"
