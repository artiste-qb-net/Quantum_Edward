import numpy as np
import numpy.random as npr
import scipy.special as sp
import scipy.stats as ss


def bin_vec_to_dec(bin_vec, nsam=1):
    """
    This function takes a 1 dim array of zeros and ones and returns the
    decimal representation of that array. If nsam > 1, the function operates
    on each row, assuming bin_vec is 2 dim matrix with bin_vec.shape[0] = nsam

    Parameters
    ----------
    bin_vec : np.array
        entries of array are either 0 or 1
    nsam : int
        number of samples

    Returns
    -------
    int | np.array

    """

    def fun(bv):
        size = len(bv)
        x = np.sum(np.array(
            [bv[k]*(1 << (size-1-k)) for k in range(size)]
        ))
        return x

    if nsam == 1:
        return fun(bin_vec)
    elif nsam > 1:
        return np.array([fun(bin_vec[sam, :]) for sam in range(nsam)])
    else:
        assert False, "illegal value for number of samples"


def dec_to_bin_vec(dec, size, nsam=1):
    """
    This function takes in an int and it returns a 1 dim array with entries
    0 or 1 and length equal to 'size'. It checks to make sure 'size' is
    large enough to fit the input. If nsam > 1, the function operates on
    each entry, assuming dec is 1 dim matrix with dec.shape[0] = nsam

    Parameters
    ----------
    dec : int | np.array
    size : int
    nsam : int
        number of samples

    Returns
    -------
    np.array
        entries of array are either 0 or 1

    """
    def fun(scalar):
        assert scalar < (1 << size), "size " + str(size) + " is too small"\
        " to fit bin rep of " + str(scalar)
        return np.array([(scalar >> k) & 1 for k in range(size)])
    if nsam == 1:
        return fun(dec)
    elif nsam > 1:
        return np.stack([fun(dec[sam]) for sam in range(nsam)])
    else:
        assert False, "illegal value for number of samples"


def ang_to_cs2_prob(ang, use_sin=1):
    """
    This function takes a float 'ang' and returns its cosine or sine
    squared, depending on flag 'use_sin'. The function also works if ang is
    a numpy array, in which case it acts elementwise and returns an array of
    the same shape as 'ang'.

    Parameters
    ----------
    ang : float | np.array
    use_sin : int
        Either 0 or 1

    Returns
    -------
    float | np.array

    """

    if use_sin == 1:
        fun = np.sin
    elif use_sin == 0:
        fun = np.cos
    else:
        assert False
    return np.square(fun(ang))


def log_beta_prob(x, conc0, conc1):
    """
    This function takes in a probability 'x' and returns the log probability
    of 'x', according to the Beta distribution with concentrations 'conc0'
    and 'conc1'. The Wikipedia article cited below refers to conc0 as alpha
    and to conc1 as beta.

    x, conc0, conc1, g0 and g1 are all floats, or they are all numpy arrays
    of the same shape. This method works elementwise for arrays.

    References
    ----------
    https://en.wikipedia.org/wiki/Beta_distribution

    Parameters
    ----------
    x : float | np.array
    conc0 : float
        Concentration 0, must be >= 0
    conc1 : float | np.array
        Concentration 1, must be >= 0

    Returns
    -------
    float | np.array

    """
    x = np.clip(x, .00001, .99999)
    return np.log(ss.beta.pdf(x, conc0, conc1))


def grad_log_beta_prob(x, conc0, conc1):
    """
    This function takes in a probability 'x' and returns the GRADIENT of the
    log probability of 'x', according to the Beta distribution with
    concentrations 'conc0' and 'conc1'. The Wikipedia article cited below
    refers to conc0 as alpha and to conc1 as beta.

    x, conc0, conc1, g0, g1 are all floats, or they are all numpy arrays of
    the same shape. This method works elementwise for arrays.

    References
    ----------
    https://en.wikipedia.org/wiki/Beta_distribution


    Parameters
    ----------
    x : float | np.array
        x \in interval [0, 1]
    conc0 : float | np.array
        Concentration 0, must be >= 0
    conc1 : float | np.array
        Concentration 1, must be >= 0

    Returns
    -------
    [g0, g1]: list[float | np.array, float | np.array]
        g0 and g1 are tha gradients with respect to the concentrations conc0
        and conc1.

    """
    def fun(z):
        a = np.clip(np.real(z), 1e-5, 15)
        b = np.clip(np.imag(z), 1e-5, 15)
        return sp.digamma(a+b) - sp.digamma(a)
    vfun = np.vectorize(fun)
    x = np.clip(x, .00001, .99999)
    g0 = np.log(x) + vfun(conc0 + 1j*conc1)
    g1 = np.log(1-x) + vfun(conc1 + 1j*conc0)
    return [g0, g1]


def log_bern_prob(x, prob1):
    """
    This function takes in x= 0 or 1 and returns the log probability of 'x',
    according to the Bernoulli distribution. So it returns P(x=0)= 1-prob1,
    P(x=1)= prob1

    x, prob1 and output are all scalars, or they are all numpy arrays of the
    same shape. This method works elementwise for arrays.

    References
    ----------

    https://en.wikipedia.org/wiki/Bernoulli_distribution

    Parameters
    ----------
    x : int | np.array
        either 0 or 1
    prob1 : float | np.array
        probability in closed interval [0, 1]

    Returns
    -------
    float | np.array

    """
    prob1 = np.clip(prob1, .00001, .99999)
    return np.log(ss.bernoulli.pmf(x, prob1))


def av_each_elem_in_array_list(list1):
    """
    'list1' is a list of numpy arrays of possibly different shapes. This
    function takes 'list1' and returns a new list which replaces each entry
    of 'list1' by its average (a scalar).

    Parameters
    ----------
    list1 : list[np.array]

    Returns
    -------
    list[float]

    """
    return np.array([np.mean(x) for x in list1])


def get_shapes_of_array_list(list1):
    """
    'list1' is a list of numpy arrays of possibly different shapes. This
    function takes 'list1' and returns a new list which replaces each entry
    of 'list1' by its shape.


    Parameters
    ----------
    list1 : list[np.array]

    Returns
    -------
    list[tuples]

    """
    return [arr.shape for arr in list1]


def new_uniform_array_list(val, shapes):
    """
    This function returns a list whose kth entry is an array with shape=
    shapes[k] and with all entries equal to 'val'.

    Parameters
    ----------
    val : float
    shapes : list[tuples]

    Returns
    -------
    list[np.array]

    """
    return [np.ones(shapes[k])*val for k in range(len(shapes))]


def new_uniformly_random_array_list(low, high, shapes):
    """
    This function returns a list whose kth entry is an array with shape=
    shapes[k] and with entries selected uniformly at random from
    the interval [low, high].

    Parameters
    ----------
    low : float
    high : float
    shapes : list[tuples]

    Returns
    -------
    list[np.array]

    """
    return [npr.uniform(low, high, size=shapes[k])
             for k in range(len(shapes))]

if __name__ == "__main__":
    def main():
        vec = dec_to_bin_vec(10, 5)
        dec = bin_vec_to_dec(vec)
        print("10, vec, dec\n", 10, vec, dec)
        dec_to_bin_vec(10, 4)
        # uncomment this to see that assert on size works
        # dec_to_bin_vec(10, 3)
    main()
