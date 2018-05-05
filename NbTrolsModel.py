import numpy as np
import numpy.random as npr
import scipy.stats as ss
import utilities as ut
import math
from Model import *


class NbTrolsModel(Model):
    """
    In this class and its twin class NoNbTrolsModel, we consider Quantum
    circuits of the following kind. Below we represent them in Qubiter ASCII
    picture notation in ZL convention, for nb=3 and na=4

    [--nb---]   [----na-----]

    NbTrols (nb Controls) model:
    0   0   0
    |---|---Ry--%---%---%---%
    |---Ry--%---%---%---%---%
    Ry--%---%---%---%---%---%
    M   M   M

    NoNbTrols (no nb Controls) model:
    0   0   0
    |---|---Ry--%---%---%---%
    |---Ry--|---%---%---%---%
    Ry--|---|---%---%---%---%
    M   M   M

    A gate |---|---Ry--%---%---%---% is called an MP_Y Multiplexor,
    or plexor for short. In the Qubiter repo at github, see Rosetta Stone
    pdf and Quantum CSD Compiler folder for more info about multiplexors.

    In NbTrols and NoNbtrols models, each layer of a list1 corresponds to a
    single plexor. We list plexors (layers) in a list1 in decreasing order
    of distance between Ry's target qubit and 0th qubit.


    References
    ----------
    1. https://github.com/artiste-qb-net/qubiter
    2. "Quantum Edward Algebra.pdf", pdf in this repo

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
        return [(1 << (na + nb - 1 - k),) for k in range(nb)]

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
        verbose : str

        Returns
        -------
        float

        """
        na =  self.na
        nb = self.nb
        powna = 1 << na
        pownb = 1 << nb
        prob = 1.

        # y = output in decimal
        ybin = ut.dec_to_bin_vec(y, nb)
        # x = input in decimal
        xbin = ut.dec_to_bin_vec(x, na)
        num_nb_trols = nb - 1
        for angs in list1_angs:
            ylast_bit = ybin[num_nb_trols]
            if num_nb_trols == 0:
                x_yrest = x
            else:
                x_yrest = ut.bin_vec_to_dec(np.concatenate(
                    [xbin, ybin[: num_nb_trols]]))
            factor = ut.ang_to_cs2_prob(angs[x_yrest], ylast_bit)
            if verbose:
                print('num_nb_trols=', num_nb_trols,
                      "x_yrest, ylast_bit=",
                      ut.dec_to_bin_vec(x_yrest, na + num_nb_trols), ylast_bit,
                      "factor=", factor)
            prob *= factor

            num_nb_trols -= 1

        return prob


if __name__ == "__main__":
    def main():
        nsam = 10
        na = 2
        nb = 3
        mod = NbTrolsModel(nb, na)
        y_nsam_nb, x_nsam_na = mod.gen_toy_data(nsam, verbose=True)
        print("x_nsam_na:")
        print(x_nsam_na)
        print("\ny_nsam_nb:")
        print(y_nsam_nb)
    main()
