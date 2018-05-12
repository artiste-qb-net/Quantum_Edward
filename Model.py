import numpy as np
import numpy.random as npr
import scipy.stats as ss
import utilities as ut


class Model:
    """
    The models (objects of this class) represent quantum circuits that
    operate on na + nb qubits. The input state goes into the first na
    qubits. The other nb qubits receive |0> for input. The output of the
    first na qubits is not measured. Only the last nb qubits are measured at
    the output.

    We will call a list1 any list of numpy arrays of shapes given by the
    list self.shapes1, with self.len1 = len(self.shapes1). len1 will be
    referred to as the number of layers.

    The first version of Quantum Edward considers two models called NbTrols
    and NoNbTrols, but QEdward is more general than those. It was written so
    that it can also analyze other akin models. Akin models should have len1
    'layers'.

    len1=nb for both models NbTrols and NoNbTrols, but they have different
    shapes1. In fact, for the NoNbTrols model

    self.shapes1 =[(powna,)]*nb

    where powna = 2^na = 1 << na, whereas for the NbTrols model,

    self.shapes1 =[(1 << (na +  k),) for k in range(nb)]

    The Wikipedia article cited below refers to the Beta distribution
    concentrations conc0 and conc1 as alpha and beta, respectively.

    References
    ----------
    https://en.wikipedia.org/wiki/Beta_distribution

    """

    def __init__(self, nb, na, list1_conc0_prior=None,
                     list1_conc1_prior=None):
        """
        Constructor

        Parameters
        ----------
        nb : int
            Total number of qubits is na + nb. Input goes into first na qubits
            and only last nb measured
        na : int
        list1_conc0_prior : list[np.array]
            a list1 of conc0 for a beta distribution used as priors
        list1_conc1_prior : list[np.array]
            a list1 of conc1 for a beta distribution used as priors

        Returns
        -------
        None

        """
        self.na = na
        self.nb = nb
        
        self.shapes1 = self.get_shapes1()
        self.len1 = len(self.shapes1)
        # print("..", self.len1)

        self.list1_conc0_prior = list1_conc0_prior
        self.list1_conc1_prior = list1_conc1_prior
        if list1_conc0_prior is None or list1_conc1_prior is None:
            self.list1_conc0_prior = ut.new_uniformly_random_array_list(
                low=0., high=5., shapes=self.shapes1)
            self.list1_conc1_prior = ut.new_uniformly_random_array_list(
                low=0., high=5., shapes=self.shapes1)

        # with beta function, if two concentrations are equal,
        # mean_x beta(x, conc0=1, conc1=1) = .5
        # ang = prob*dpi
        self.list1_angs_prior = [ss.beta.mean(self.list1_conc0_prior[k],
                                 self.list1_conc1_prior[k])*ut.dpi
                                 for k in range(self.len1)]

    def get_shapes1(self):
        """
        Abstract method. Must be overridden by descendant class. In
        descendant class, this function should return a list of the shapes
        of the elements of a list1.

        Returns
        -------
        list[tuple]

        """
        assert False, "this function must be overridden"

    def prob_y_for_given_x_and_angs_prior(self, y, x,
                                          list1_angs,
                                          verbose=False):
        """
        Abstract method. Must be overridden by descendant class. In
        descendant class, should return the probability of y given x and
        list1_angs. P(y | x, list1_angs)

        The Bayesian net for the model looks like this

        x    list1_angs
        |  /
        y

        with the arrows pointing downward.

        y is an int in range(pownb). x is an int in range(powna). list1_angs
        is a list1 of numpy arrays containing qc qubit rotation angles.

        y and x are observed variables. nsam measurements of (y, x) are
        given by the training data. list1_angs are hidden variables. We want
        to estimate those values for list1_angs which best fit the training
        data.

        Parameters
        ----------
        y : int
        x : int
        list1_angs : list[np.array]
        verbose : bool

        Returns
        -------
        float

        """
        assert False, "this function must be overridden"

    def sum_over_y_of_prob_y(self, x, list1_angs):
        """
        This function should return 1. It can be used to check that the
        function prob_y_for_given_x_and_angs_prior() yields a set of
        probabilities which when summed over y gives 1.

        Parameters
        ----------
        x : int
        list1_angs : list[np.array]

        Returns
        -------
        float

        """
        pownb = 1 << self.nb
        tot_prob = 0.
        for y in range(pownb):
            tot_prob == self.prob_y_for_given_x_and_angs_prior(y, x,
                                          list1_angs)
        return tot_prob

    def gen_toy_data(self, nsam, verbose=False):
        """
        This function generates nsam samples of (y, x). It returns these
        samples as arrays with 0 and 1 entries y_nsam_nb and x_nsam_nb.
        y_nsam_nb.shape = (nsam, nb), x_nsam_na.shape = (nsam, na).

        For example, x_nsam_na[0, :] could look like [1, 0, 1, 1], meaning
        that upon input with na=4 and nb=3, the qubit values are

        q0=1
        q1=0
        q2=1
        q3=1
        q4=0
        q5=0
        q6=0

        Parameters
        ----------
        nsam : int
        verbose : bool

        Returns
        -------
        y_nsam_nb, x_nsam_na : tuple[np.array, np.array]

        """

        # x_nsam_na entries are integers 0 or 1
        x_nsam_na = npr.randint(low=0, high=2, size=(nsam, self.na))
        # print(x_nsam_na)

        # x_nsam entries are integers in range(powna)
        x_nsam = ut.bin_vec_to_dec(x_nsam_na, nsam=nsam)
        # print(x_nsam)

        # p_nsam_pownb entries will be probabilities
        pownb = 1 << self.nb
        p_nsam_pownb = np.zeros((nsam, pownb))
        for sam in range(nsam):
            p_nsam_pownb[sam, :] = np.array([
                self.prob_y_for_given_x_and_angs_prior(y, x_nsam[sam],
                                               self.list1_angs_prior,
                                               verbose)
                for y in range(pownb)])
            if verbose:
                print("sample=", sam,
                      "  total_prob=", np.sum(p_nsam_pownb[sam, :]),
                      '\n')

        # multinomial with just one draw will give vec with
        # all entries 0 except one entry equal to 1
        y_nsam = np.array(
            [list(npr.multinomial(n=1, pvals=p_nsam_pownb[sam, :])).index(1)
             for sam in range(nsam)])

        y_nsam_nb = ut.dec_to_bin_vec(y_nsam, self.nb, nsam=nsam)

        return y_nsam_nb, x_nsam_na


if __name__ == "__main__":
    def main():
        print(5)
    main()
