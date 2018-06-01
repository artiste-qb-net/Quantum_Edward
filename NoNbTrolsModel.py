import numpy as np
import numpy.random as npr
import scipy.stats as ss
import utilities as ut
import math
from Model import *
from NbTrolsModel import *


class NoNbTrolsModel(Model):
    """
    See docstring describing class NbTrolsModel

    """

    def __init__(self, nb, na):
        """
        Constructor

        Parameters
        ----------
        nb : int
        na : int

        Returns
        -------
        None

        """
        Model.__init__(self, nb, na)

    def get_shapes1(self):
        """
        Returns a list of the shapes of the elements of a list1.

        Returns
        -------
        list[tuple]

        """
        na = self.na
        nb = self.nb
        powna = 1 << na
        return [(1 << k,) for k in range(na)] + [(powna,)]*nb

    def prob_x(self, x,
               list1_angs,
               verbose=False):
        """
        Returns probability of input x, P(x).

        Parameters
        ----------
        x : int
            x is an int in range(powna).
        list1_angs : list[np.array]
        verbose : bool

        Returns
        -------
        float

        """
        return NbTrolsModel.static_prob_x(x, self.na, list1_angs, verbose)

    def prob_y_given_x_and_angs_prior(self, y, x,
                                      list1_angs,
                                      verbose=False):
        """
        Returns the probability of y given x and list1_angs, P(y | x,
        list1_angs).

        Parameters
        ----------
        y : int
            int in range(pownb)
        x : int
            int in range(powna)
        list1_angs : list[np.array]
        verbose : bool

        Returns
        -------
        float

        """
        na = self.na
        prob = 1.

        # y = output in decimal
        ybin = ut.dec_to_bin_vec(y, self.nb)
        num_trols = na
        for angs in list1_angs[na:]:
            ylast_bit = ybin[num_trols-na]
            factor = ut.ang_to_cs2_prob(angs[x], ylast_bit)
            if verbose:
                print('num_trols=', num_trols,
                      "ybin[:num_trols-na], y_last_bit",
                      ybin[:num_trols-na],
                      ylast_bit)
            prob *= factor

            num_trols += 1

        if verbose:
            print('\tybin, prob:', ybin, prob)

        return prob


if __name__ == "__main__":
    def main():
        nsam = 10
        na = 2
        nb = 3
        mod = NoNbTrolsModel(nb, na)
        y_nsam_nb, x_nsam_na = mod.gen_toy_data(nsam, verbose=True)
        print("x_nsam_na:")
        print(x_nsam_na)
        print("\ny_nsam_nb:")
        print(y_nsam_nb)
    main()
