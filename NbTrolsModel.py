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
    |0> |0> |0> |0> |0> |0> |0>
    NOTA P(x) next
    |---|---|---|---|---|---Ry
    |---|---|---|---|---Ry--%
    |---|---|---|---Ry--%---%
    |---|---|---Ry--%---%---%
    NOTA P(y|x) next
    |---|---Ry--%---%---%---%
    |---Ry--%---%---%---%---%
    Ry--%---%---%---%---%---%
    M   M   M

    NoNbTrols (no nb Controls) model:
    |0> |0> |0> |0> |0> |0> |0>
    NOTA P(x) next
    |---|---|---|---|---|---Ry
    |---|---|---|---|---Ry--%
    |---|---|---|---Ry--%---%
    |---|---|---Ry--%---%---%
    NOTA P(y|x) next
    |---|---Ry--%---%---%---%
    |---Ry--|---%---%---%---%
    Ry--|---|---%---%---%---%
    M   M   M

    A gate |---|---Ry--%---%---%---% is called an MP_Y Multiplexor,
    or plexor for short. In Ref.1 (Qubiter repo at github), see Rosetta Stone
    pdf and Quantum CSD Compiler folder for more info about multiplexors.

    In NbTrols and NoNbtrols models, each layer of a list1 corresponds to a
    single plexor. We list plexors (layers) in a list1 in order of
    increasing distance between the Ry target qubit and the 0th qubit.

    Note that the expansion of a multiplexor into elementary gates (cnots
    and single qubit rotations) contains a huge number of gates (exp in the
    number of controls). However, such expansions can be shortened by
    approximating the multiplexors, using, for instance, the technique of
    Ref.2.

    Ref.3 explains the motivation for choosing this model. This model is in
    fact guaranteed to fully parametrize P(x) and P(y|x).

    The circuits given above are for finding a fit of both P(x) and P(y|x).
    However, if one wants to use a physical hardware device as a classifier,
    then one should omit the beginning part of the circuits (the parts that
    represent P(x)), and feed the input x into the first na qubits. In other
    words, for classifying, use the following circuits instead of the ones
    above:

    [--nb---]   [----na-----]

    NbTrols (nb Controls) model:
    |0> |0> |0>
    |---|---Ry--%---%---%---%
    |---Ry--%---%---%---%---%
    Ry--%---%---%---%---%---%
    M   M   M

    NoNbTrols (no nb Controls) model:
    |0> |0> |0>
    |---|---Ry--%---%---%---%
    |---Ry--|---%---%---%---%
    Ry--|---|---%---%---%---%
    M   M   M


    References
    ----------
    1. https://github.com/artiste-qb-net/qubiter

    2. Oracular Approximation of Quantum Multiplexors and Diagonal Unitary
    Matrices, by Robert R. Tucci, https://arxiv.org/abs/0901.3851

    3. Code Generator for Quantum Simulated Annealing, by Robert R. Tucci,
    https://arxiv.org/abs/0908.1633 , Appendix B

    4. "Quantum Edward Algebra.pdf", pdf included in this repo

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
        return [(1 << k,) for k in range(na+nb)]

    @staticmethod
    def static_prob_x(x, na, list1_angs, verbose=False):
        """
        Returns probability of input x, P(x).

        Parameters
        ----------
        x : int
            x is an int in range(powna).
        na : int
        list1_angs : list[np.array]
        verbose : bool

        Returns
        -------
        float

        """
        prob = 1.

        # x = input in decimal
        xbin = ut.dec_to_bin_vec(x, na)
        num_trols = 0
        # print(".,.", list1_angs)
        for angs in list1_angs[0:na]:
            xlast_bit = xbin[num_trols]
            if num_trols == 0:
                xrest = 0
                angs1 = float(angs)
            else:
                xrest = ut.bin_vec_to_dec(xbin[: num_trols])
                angs1 = angs[xrest]
            factor = ut.ang_to_cs2_prob(angs1, xlast_bit)
            if verbose:
                print('num_trols=', num_trols,
                      "xrest_bin, xlast_bit=",
                      xbin[: num_trols],
                      xlast_bit)
            prob *= factor

            num_trols += 1
        if verbose:
            print("\txbin, prob:", xbin, prob)

        return prob

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
        verbose : str

        Returns
        -------
        float

        """
        na = self.na
        nb = self.nb
        prob = 1.

        # y = output in decimal
        ybin = ut.dec_to_bin_vec(y, nb)
        # x = input in decimal
        xbin = ut.dec_to_bin_vec(x, na)
        num_trols = na
        for angs in list1_angs[na:]:
            ylast_bit = ybin[num_trols-na]
            if num_trols == na:
                x_yrest_bin = x
                x_yrest = x
            else:
                x_yrest_bin = np.concatenate([xbin,
                                              ybin[: num_trols-na]])
                x_yrest = ut.bin_vec_to_dec(x_yrest_bin)
            factor = ut.ang_to_cs2_prob(angs[x_yrest], ylast_bit)
            if verbose:
                print('num_trols=', num_trols,
                      "x_yrest_bin, ylast_bit=",
                      x_yrest_bin,
                      ylast_bit)
            prob *= factor

            num_trols += 1

        if verbose:
            print('\txbin, ybin, prob:', xbin, ybin, prob)

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
