"""
.. module:: odf
    :platform: Unix, Windows
    :synopsis: Module contains functions for computing and manipulating orientation distribution data

.. moduleauthor:: Sam Potter <spotter1642@gmail.com>
"""

from . import np
from . import beta


def Ifiber(a0, a2, a4, phi, theta):

    """
    Compute the double cosine series solution to Mie scattering of cylindrical fibers.

    :param a0: a0 parameter
    :type a0: float
    :param a2: a2 parameter
    :type a2: float
    :param a4: a4 parameter
    :type a4: float
    :param phi: preferred fiber direction in degrees
    :type phi: float
    :param theta: Values of theta at which to evaluate the cosine series. Values in degrees
    :type theta: ndarray
    :return: Intensity values
    :rtype: ndarray
    """

    vals = a0 + a2 * np.cos(np.deg2rad(2 * (theta - phi))) + a4 * np.cos(np.deg2rad(4 * (theta - phi)))

    return vals


def syntheticIdist(a0, a2, a4, phi, theta, splay, nsamples, distribution='uniform'):

    """
    Function for generating a synthetic pSFDI signal from a distribution of fibers directions about a mean of phi.

    :param a0: a0 parameter
    :type a0: float
    :param a2: a2 parameter
    :type a2: float
    :param a4: a4 parameter
    :type a4: float
    :param phi: preferred fiber direction in degrees
    :type phi: float
    :param theta: Values of theta at which to evaluate the cosine series. Values in degrees
    :type theta: ndarray
    :param splay: standard deviation of fiber direction distribution
    :type splay: float
    :param nsamples: number of samples to draw from the statistical distribution
    :type nsamples: int
    :param distribution: Optional. Specify type of distribution to draw samples from. Default is uniform
    :type distribution: str
    :return: Return a periodic signal representing the summed contributions of each single fiber sample to the over all
    signal
    :rtype: ndarray
    """

    Idist = np.zeros((nsamples, len(theta)))
    for i in range(0, nsamples):
        if distribution == 'uniform':
            phi = np.random.uniform(-1 * splay, splay)
        elif distribution == 'normal':
            phi = np.random.normal(0, splay)

        vals = Ifiber(a0, a2, a4, phi, theta)
        Idist[i, :] = vals

    Idist = np.sum(Idist, axis=0) / nsamples

    return Idist


def distribution_minimand(a0, a2, a4, phi, theta, data):

    """
    Function that compares a trial vector of ai cosine series parameters with given data and returns the sum of square
    differences between the quantities

    :param a0: a0 parameter
    :type a0: float
    :param a2: a2 parameter
    :type a2: float
    :param a4: a4 parameter
    :type a4: float
    :param phi: preferred fiber direction in degrees
    :type phi: float
    :param theta: Values of theta at which to evaluate the cosine series. Values in degrees
    :type theta: ndarray
    :param data: Cosine series data against which the model is being compared. Must be same length as theta array
    :type data: ndarray
    :return: Sum of squared differences between cosine series using given ai parameters and data
    :rtype: float
    """

    feval = Ifiber(a0, a2, a4, phi, theta)

    diff = data - feval

    diffsq = np.square(diff, diff)

    ssd = np.sum(diffsq)

    return ssd


def distribution_minfun(params, *args):

    """
    Function for passing to scipy optimization algorithm that calls the distribution minimand. Used to find ai terms to
    fit a distribution of fibers.

    :param params: ai terms and phi in a list like structure
    :type: ndarray
    :param args: tuple containing additional needed data for the minimand
    :type args: tup`
    :return: evaluated ssd via the minimand
    :rtype: float
    """

    return distribution_minimand(params[0], params[1], params[2], params[3], *args)


def compute_beta(mean, sd, theta):

    """
    This function computes the value of a beta distribution with mean and standard deviation (mean, sd) at location
    theta on the interval [-pi/2, pi/2]

    :param mean: Mean of beta distribution
    :type mean: float
    :param sd: Standard deviation of beta distribution
    :type sd: float
    :param theta: Argument of Beta distribution pdf. Can be array.
    :type theta: ndarray
    :return: Value of specified beta distribution pdf at point theta
    :rtype: float array_like
    """

    # Map mean and standard deviation to [0, 1] range of the standard beta distribution
    mu = (mean + np.pi / 2) / np.pi
    sigma = sd / np.pi

    # Compute the shape parameters
    gamma = (mu ** 2 - mu ** 3 - sigma ** 2 * mu) / (sigma ** 2)
    delta = gamma * (1 - mu) / mu

    # Map evaluation point to [0, 1] range of the standard beta distribution
    y = (theta + np.pi / 2) / np.pi

    # Evaluate pdf
    vals = beta.pdf(y, gamma, delta) / np.pi

    return vals


def compute_gamma(mean1, sd1, mean2, sd2, theta, d=0.5):

    """
    This function computes a normalized orientation distribution by combining the results from two separate beta
    distributions with different means, standard deviations and user defined weighting factor between the distributions

    :param mean1: Mean of first beta distribution
    :type mean1: float
    :param sd1: Standard deviation of first beta distribution
    :type sd1: float
    :param mean2: Mean of second beta distribution
    :type mean2: float
    :param sd2: Standard deviation of second beta distribution
    :type sd2: float
    :param theta: Argument of Beta distribution pdf. Can be array.
    :type theta: ndarray
    :param d: Optional. Weight factor for adding up the two distributions. Default 0.5
    :type d: float
    :return: Normalized value of combined beta distributions
    :rtype: float array_like
    """

    # Adjust theta so it wraps around to the -pi/2 pi/2 interval
    local_theta = np.copy(theta)
    local_theta[local_theta < -np.pi / 2] = local_theta[local_theta < -np.pi / 2] + np.pi
    local_theta[local_theta > np.pi / 2] = local_theta[local_theta > np.pi / 2] - np.pi

    # Compute value of distribution pdfs
    gamma1 = compute_beta(mean1, sd1, local_theta)
    gamma2 = compute_beta(mean2, sd2, local_theta)

    # Combine distributions using weight factor and shift accordingly
    gamma = d * (gamma1 + gamma2) + (1 - d) / np.pi

    # Normalize the result so it's an ODF
    gamma_area = 0.0
    for i in range(1, len(theta)):
        gamma_area += 0.5 * (gamma[i] + gamma[i - 1]) * (theta[i] - theta[i - 1])

    return gamma / gamma_area
