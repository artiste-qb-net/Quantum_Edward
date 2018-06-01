import numpy as np
import numpy.random as npr
import scipy.stats as ss
import utilities as ut
from TimeStep import *
from Plotter import *


class Fitter:
    """
    Read docstrings for Model class first.

    The goal of this class is to implement the BBVI(see ref below) for a 
    Model object 'model' to estimate those values for the hidden variables
    list1_angs which best fit the training data y_nsam_nb, x_nsam_na.
    
    In BBVI, one maximizes ELBO with respect to a parameter lambda. In this
    case, lambda = list1_conc0, list1_conc1 and z = list1_z =
    list1_angs/dpi. The angles in list1_angs are in the interval [0, dpi] so
    the entries list1_z are in the interval [0, 1].
    
    References
    ----------   
    R. Ranganath, S. Gerrish, D. M. Blei, "Black Box Variational
    Inference", https://arxiv.org/abs/1401.0118

    """

    def __init__(self, model, y_nsam_nb, x_nsam_na, nsamgrad,
                 nt, eta, t_step_meth):
        """
        Constructor

        Parameters
        ----------
        model : Model
        y_nsam_nb : np.array     
            An array of zeros and ones with shape=(nsam, nb) containing nsam 
            samples of y output.             
        x_nsam_na : np.array
            An array of zeros and ones with shape=(nsam, na) containing nsam 
            samples of x input. 
        nsamgrad : int
            Number of samples used during averaging of the gradient of ELBO
        nt : int
            Number of time steps (aka iterations). Value of ELBO changes (
            increases or stays the same) with each iteration.
        eta : float
            positive scaling parameter (proportionality factor) for delta
            lambda. Passed to TimeStep class
        t_step_meth : str
            str labelling the method used to calculate delta lambda. This
            str is passed to TimeStep class.

        Returns
        -------
        None

        """
        self.mod = model
        self.y_nsam_nb = y_nsam_nb
        self.x_nsam_na = x_nsam_na
        self.nsamgrad = nsamgrad
        self.nt = nt
        self.eta = eta
        self.t_step_meth = t_step_meth

        assert self.mod.na == x_nsam_na.shape[1]
        assert self.mod.nb == y_nsam_nb.shape[1]
        assert self.y_nsam_nb.shape[0] == self.x_nsam_na.shape[0]

        # the following will be filled by do_fit()
        self.fin_t = None
        self.fin_list1_conc0 = None
        self.fin_list1_conc1 = None

        len1 = self.mod.len1
        self.conc_nt_2_len1 = np.zeros((nt, 2, len1), dtype=float)
        self.delta_conc_nt_2_len1 = np.zeros((nt, 2, len1), dtype=float)
        self.elbo_nt_len1 = np.zeros((nt, len1), dtype=float)

    def get_delbo_and_grad_delbo(self, list1_z, list1_conc0, list1_conc1):
        """
        delbo = density of elbo. grad = gradient. This is a private 
        auxiliary function used by do_fit(). Inside the method do_fit(), 
        we calculate elbo from delbo by taking expected value of delbo over 
        z~ q(z | lambda) 


        Parameters
        ----------
        list1_z : list[np.array]
        list1_conc0 : list[np.array]
        list1_conc1 : list[np.array]

        Returns
        -------
        tuple[list[np.array], list[np.array], list[np.array]]

        """
        nsam = self.y_nsam_nb.shape[0]
        len1 = self.mod.len1

        # grad0,1 log q(z| lambda=conc0, conc1)
        xx = [ut.grad_log_beta_prob(list1_z[k],
                                        list1_conc0[k],
                                        list1_conc1[k])
                                for k in range(len1)]
        # zip doesn't work
        # list1_g0, list1_g1 = zip(xx)

        def my_zip(a):
            return [[a[j][k] for j in range(len(a))]
                    for k in range(len(a[0]))]

        # print('---------xx')
        # for j in range(2):
        #     print(j, xx[j])
        # print('---------zip(zz)')
        # for j in range(2):
        #     tempo = list(zip(xx))
        #     print(j, tempo[j])
        # print('---------my_zip(zz)')
        # for j in range(2):
        #     tempo = my_zip(xx)
        #     print(j, tempo[j])

        list1_g0, list1_g1 = my_zip(xx)

        # sum_sam (log p(y| x,  z = angs/dpi))
        x_nsam = ut.bin_vec_to_dec(self.x_nsam_na, nsam=nsam)
        y_nsam = ut.bin_vec_to_dec(self.y_nsam_nb, nsam=nsam)
        list1_angs = [list1_z[k]*ut.dpi for k in range(len1)]
        # log_py is a constant with shape 1
        log_py = np.sum(np.log(1e-8 + np.array(
            [self.mod.prob_y_given_x_and_angs_prior(y_nsam[sam],
                x_nsam[sam], list1_angs) for sam in range(nsam)]
        )))

        # log_px is a constant with shape 1
        log_px = np.sum(np.log(1e-8 + np.array(
            [self.mod.prob_x(x_nsam[sam], list1_angs) for sam in range(nsam)]
        )))

        # log p(z)
        list1_log_pz = [ut.log_beta_prob(list1_z[k],
                                  self.mod.list1_conc0_prior[k],
                                  self.mod.list1_conc1_prior[k])
                        for k in range(len1)]

        # log q(z| lambda)
        list1_log_qz = [ut.log_beta_prob(list1_z[k],
                                  list1_conc0[k],
                                  list1_conc1[k])
                        for k in range(len1)]

        # log p(y, x, z) - log q(z | lambda)
        list1_delbo = [log_py + log_px + list1_log_pz[k] - list1_log_qz[k]
                 for k in range(len1)]
        # print("//", len1, "log_py=", log_py, list1_delbo)

        list1_grad0_delbo = [np.multiply(list1_g0[k], list1_delbo[k])
                       for k in range(len1)]

        list1_grad1_delbo = [np.multiply(list1_g1[k], list1_delbo[k])
                       for k in range(len1)]

        return list1_delbo, list1_grad0_delbo, list1_grad1_delbo

    def do_fit(self):
        """
        This function attempts to maximize ELBO over lambda. Does at most nt
        iterations (i.e., lambda changes, time steps). But may reach a
        convergence condition before doing nt iterations. Final iteration
        time is stored in self.fin_t.
        
        This function stores final values for time and lambda (lambda =
        concentrations 0, 1)
        
        self.fin_t
        self.fin_list1_conc0
        self.fin_list1_conc1

        It also stores traces (time series) for lambda (lambda =
        concentrations 0, 1), delta lambda between consecutive steps,
        and the ELBO value:
        
        self.conc_nt_2_len1
        self.delta_conc_nt_2_len1
        self.elbo_nt_len1

        Returns
        -------
        None

        """
        len1 = self.mod.len1

        # starting values
        shapes = self.mod.shapes1
        list1_conc0 = ut.new_uniform_array_list(1., shapes)
        list1_conc1 = ut.new_uniform_array_list(1., shapes)
        step = TimeStep(self.t_step_meth, self.eta, self.mod.len1)

        for t in range(self.nt):

            list1_elbo = ut.new_uniform_array_list(0., shapes)
            list1_grad0_elbo = ut.new_uniform_array_list(0., shapes)
            list1_grad1_elbo = ut.new_uniform_array_list(0., shapes)
            
            for s in range(self.nsamgrad):
                list1_z = [ss.beta.rvs(list1_conc0[k], list1_conc1[k])
                           for k in range(len1)]
                x0, x1, x2 =\
                    self.get_delbo_and_grad_delbo(list1_z,
                                                  list1_conc0,
                                                  list1_conc1)
                for k in range(len1):
                    list1_elbo[k] += x0[k]/self.nsamgrad
                    list1_grad0_elbo[k] += x1[k]/self.nsamgrad
                    list1_grad1_elbo[k] += x2[k]/self.nsamgrad

            g0 = list1_grad0_elbo
            g1 = list1_grad1_elbo
            for k in range(len1):
                delta_conc = step.get_delta_conc(g0[k], g1[k], t, k)

                old_conc0 = np.copy(list1_conc0[k])
                list1_conc0[k] += delta_conc[0]
                list1_conc0[k] = np.clip(list1_conc0[k], 1e-5, 15)
                true_delta_conc0 = list1_conc0[k] - old_conc0

                old_conc1 = np.copy(list1_conc1[k])
                list1_conc1[k] += delta_conc[1]
                list1_conc1[k] = np.clip(list1_conc1[k], 1e-5, 15)
                true_delta_conc1 = list1_conc1[k] - old_conc1

                self.conc_nt_2_len1[t, 0, k] = np.sum(list1_conc0[k])
                self.conc_nt_2_len1[t, 1, k] = np.sum(list1_conc1[k])
                self.delta_conc_nt_2_len1[t, 0, k] = np.sum(true_delta_conc0)
                self.delta_conc_nt_2_len1[t, 1, k] = np.sum(true_delta_conc1)

            self.elbo_nt_len1[t, :] = \
                ut.av_each_elem_in_array_list(list1_elbo)
            if np.all(self.delta_conc_nt_2_len1[t, :, :] < 0.001):
                break

        self.fin_t = t
        self.fin_list1_conc0 = list1_conc0
        self.fin_list1_conc1 = list1_conc1
        
    def print_fit_values_at_fin_t(self):
        """
        Prints to screen summary of values at final time fin_t of do_fit()
        run.

        Recall z = ang/dpi with ang in interval [0, dpi] so z in interval [
        0, 1].This function calculates for each z, its estimate, the std of
        that estimate, and the fractional error (z_estimate -
        z_prior)/z_prior. z_prior = angs_prior/dpi.

        angs_prior are the prior angles assumed for the model. If we use
        training data generated by Model:get_toy_data(), angs_prior are true
        values, the ones used to generate the synthetic data.

        Returns
        -------
        None

        """
        len1 = self.mod.len1
        list1_conc0 = self.fin_list1_conc0
        list1_conc1 = self.fin_list1_conc1

        list1_zpred = [ss.beta.mean(list1_conc0[k], list1_conc1[k])
                       for k in range(len1)]
        list1_std_zpred = [ss.beta.std(list1_conc0[k], list1_conc1[k])
                           for k in range(len1)]

        print('fin_t=', self.fin_t, "\n")
        for k in range(len1):
            print("list1_z[" + str(k) + "]:")
            print("estimate:\n" + str(list1_zpred[k]))
            print("st.dev.:\n" + str(list1_std_zpred[k]))
            zprior = self.mod.list1_angs_prior[k]/ut.dpi
            print("frac. error = (est-prior)/prior:\n" +
                  str((list1_zpred[k] - zprior)/zprior) + "\n")

    def plot_fit_traces(self):
        """
        Calls Plotter to plot traces (time series) collected during do_fit() 
        run. Plots time series of lambda, delta lambda and ELBO. 

        Returns
        -------
        None

        """
        Plotter.plot_conc_traces(self.fin_t,
                                 self.conc_nt_2_len1,
                                 self.delta_conc_nt_2_len1)
        Plotter.plot_elbo_traces(self.fin_t,
                                 self.elbo_nt_len1)


if __name__ == "__main__":
    from NbTrolsModel import *
    from NoNbTrolsModel import *

    def main():

        # Ridiculously small numbers,
        # just to make sure it runs without crashing
        npr.seed(1234)
        na = 2  # number of alpha qubits
        nb = 2  # number of beta qubits
        mod = NbTrolsModel(nb, na)
        # mod = NoNbTrolsModel(nb, na)

        nsam = 20  # number of samples
        y_nsam_nb, x_nsam_na = mod.gen_toy_data(nsam)

        nsamgrad = 10  # number of samples for grad estimate
        nt = 20  # number of interations

        # t_step_type, eta = naive', .0003  # very sensitive to eta
        # t_step_type, eta = 'naive_t', .0003  # very sensitive to eta
        # t_step_type, eta = 'mag1_grad', .2
        t_step_meth, eta = 'ada_grad', .1

        ff = Fitter(mod, y_nsam_nb, x_nsam_na,
                    nsamgrad, nt, eta, t_step_meth)
        ff.do_fit()
        ff.print_fit_values_at_fin_t()
        ff.plot_fit_traces()

    main()
