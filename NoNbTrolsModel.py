import numpy as np
import numpy.random as npr
import scipy.stats as ss
import utilities as ut
import math
from Model import *


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
        return [(powna,)]*nb

    def prob_y_for_given_x_and_angs_prior(self, y, x,
                                          list1_angs,
                                          verbose=False):
        """
        Returns the probability of y given x and list1_angs. P(y | x,
        list1_angs)

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
        prob = 1.

        # y = output in decimal
        ybin = ut.dec_to_bin_vec(y, self.nb)
        num_nb_trols = 0
        for angs in list1_angs:
            ylast_bit = ybin[num_nb_trols]
            factor = ut.ang_to_cs2_prob(angs[x], ylast_bit)
            if verbose:
                print('na+num_nb_trols=', self.na+num_nb_trols,
                      "ybin[:num_nb_trols+1]=", ybin[:num_nb_trols+1],
                      "ylast_bit=", ylast_bit,
                      "factor=", factor)
            prob *= factor

            num_nb_trols += 1

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
