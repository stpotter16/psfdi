"""
.. module:: odf
    :platform: Unix, Windows
    :synopsis: Module contains functions for computing and manipulating orientation distribution data

.. moduleauthor:: Sam Potter <spotter1642@gmail.com>
"""

from . import np
from . import beta
from . import interp1d
from . import fftconvolve
from . import trapz


def IfiberDeg(a0, a2, a4, phi, theta):

    """
    Compute the double cosine series solution to Mie scattering of cylindrical fibers with degree inputs

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


def syntheticIdistNormal(a0, a2, a4, phi, theta, splay, nsamples, distribution='uniform'):

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

        vals = IfiberDeg(a0, a2, a4, phi, theta)
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

    feval = IfiberDeg(a0, a2, a4, phi, theta)

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


def compute_an(order, gamma, theta):

    """
    This function computes the cosine coefficients of a Fourier series approximation to data gamma evaluated at
    independent variable values theta. Fourier series coefficients are computed up to specified order

    :param order: Order of Fourier series approximation
    :type order: int
    :param gamma: Dependent variable of data series to fit
    :type gamma. ndarray
    :param theta: Independent variable of data series to fit
    :type theta: ndarray
    :return: Cosine coefficient of specified order
    :rtype: float
    """

    #TODO-remove for loop
    coeff = 0.0
    for i in range(1, len(theta)):
        coeff += 0.5 * (gamma[i] * np.cos(order * theta[i]) + gamma[i - 1] * np.cos(order * theta[i - 1])) * (
                    theta[i] - theta[i - 1])

    return 2 * coeff


def compute_bn(order, gamma, theta):

    """
    This function computes the sine coefficients of a Fourier series approximation to data gamma evaluated at
    independent variable values theta. Fourier series coefficients are computed up to specified order

    :param order: Order of Fourier series approximation
    :type order: int
    :param gamma: Dependent variable of data series to fit
    :type gamma. ndarray
    :param theta: Independent variable of data series to fit
    :type theta: ndarray
    :return: Sine coefficient of specified order
    :rtype: float
    """

    #TODO-remove for loop
    coeff = 0.0
    for i in range(1, len(theta)):
        coeff += 0.5 * (gamma[i] * np.sin(order * theta[i]) + gamma[i - 1] * np.sin(order * theta[i - 1])) * (
                    theta[i] - theta[i - 1])

    return 2 * coeff


def fit_fourier(max_order, gamma, theta):

    """
    This function computes Fourier series coefficients for an approximation to the given (theta, gamma) data series up
    to the specified maximum order

    :param max_order: Maximum order of Fourier series approximation
    :type max_order: int
    :param gamma: Dependent variable of data series to fit
    :type gamma. ndarray
    :param theta: Independent variable of data series to fit
    :type theta: ndarray
    :return: Tuple containing coefficients (an, bn, c). C is the leading constant coefficient.
    :rtype: tuple
    """
    # Compute an
    an = [compute_an(order, gamma, theta) for order in range(2, max_order, 2)]

    # Compute bn
    bn = [compute_bn(order, gamma, theta) for order in range(2, max_order, 2)]

    # Compute c
    c = 0.0
    for i in range(1, len(theta)):
        c += 0.5 * (gamma[i] + gamma[i - 1]) * (theta[i] - theta[i - 1])

    return an, bn, c


def compute_fourier(an, bn, c, theta):

    """
    This function evaluates a Fourier series approximation at given independent variable theta

    :param an: Cosine coefficients
    :type an: list like
    :param bn: Sine coefficients
    :type bn: list like
    :param c: Constant coefficient
    :type c: float
    :param theta: Independent variable
    :type theta: ndarray
    :return: Array of evaluated Fourier seriesA
    :rtype: ndarray
    """

    # Pre-allocate
    computed = np.zeros(len(theta))

    # Compute series
    series_sum = 0.0
    for i in range(0, len(theta)):
        for j in range(0, len(an)):
            series_sum += an[j] * np.cos(2 * (j + 1) * theta[i]) + bn[j] * np.sin(2 * (j + 1) * theta[i])
        computed[i] = c / (2 * np.pi) * (1 + series_sum)
        series_sum = 0.0

    return computed


def structural_eigenval_thetas(a1, b1):

    """
    Function forms the 2nd order structural tensor from the a1 and b1 Fourier series fit terms, computes the
    eigenvalues, then computes the angle of the associated eigenvectors

    :param a1: First cosine coefficient from Fourier series fit
    :type a1: float
    :param b1: First sine coefficient from Fourier series fit
    :type b1: float
    :return: tuple of first and second eigenvector angles
    :rtype: tuple
    """

    # For structural tensor
    D2 = np.array([[a1, b1],
                   [b1, -a1]])

    # Compute the eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eig(D2)

    # Compute the eigenvector angles
    theta1 = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
    theta2 = np.arctan2(eigvecs[1, 1], eigvecs[0, 1])

    return np.rad2deg(theta1), np.rad2deg(theta2)


def ellipse(b, e, phi, theta):

    """
    Compute the value of an ellipse with specified radius, eccentricity, and angular rotation at specified angle

    :param b: Ellipse minor axis radius. Value in the interval [0, inf]
    :type b: float
    :param e: Ellipse eccentricity. Bounded value in the interval [0, 1]
    :type e: float
    :param phi: Angular offset of ellipse major axis. Radian value in interval [0, 2pi]
    :type phi: float
    :param theta: Angular argument of ellipse. Radian value in interval [0, 2pi]
    :type theta: float
    :return: Radius of ellipse.
    :rtype: float
    """

    rval = b / np.sqrt(1 - (e * np.cos(theta - phi)) ** 2)

    return rval


def minellipse(b, e, phi, theta, signal):

    """
    Minimand for computing the sum of squared differences between a fit ellipse and ellipsoidal data. Used for fitting
    ellipse model

    :param b: Ellipse minor axis radius. Value in the interval [0, inf]
    :type b: float
    :param e: Ellipse eccentricity. Bounded value in the interval [0, 1]
    :type e: float
    :param phi: Angular offset of ellipse major axis. Radian value in interval [0, 2pi]
    :type phi: float
    :param theta: Array of angular argument of ellipse. Radian values in interval [0, 2pi]
    :type theta: ndarray
    :param signal: Ellipsoidal data to compute the sum of squared differences against.
    :type signal: ndarray
    :return: Sum of square difference between data and proposed ellipse.
    :rtype: float
    """

    ellipsepts = ellipse(b, e, phi, theta)

    diff = signal - ellipsepts

    diffsq = np.square(diff, diff)

    ssd = np.sum(diffsq)

    return ssd


def minellipsefun(params, *args):

    """
    Wrapper function for ellipse fitting minimand. This function is to be passed to Scipy's minimization functions

    :param params: List of minimization parameters: b, e, phi
    :type params: list, ndarray
    :param args: Tuple containing necessary arguments to ellipse minimand: (Thetas, Signal)
    :type args: tuple
    :return: Sum of square difference between data and proposed ellipse.
    :rtype: float
    """
    return minellipse(params[0], params[1], params[2], *args)


def wrapped_normal(mu, sigma, period, thetas):

    """
    Compute a wrapped normal distribution with specified mean and standard deviation. Also allows specification of the
    wrapping period used to generate the wrapped pdf.

    :param mu: Mean of wrapped normal distribution. Radian value in interval [-pi/2, pi/2]
    :type mu: float
    :param sigma: Standard deviation of wrapped normal distribution. Radian value in interval [0, pi/2]
    :type sigma: float
    :param period: Period of shift used to compute the sum that generates the wrapped pdf.
    :type period: float
    :param thetas: Radian angular values at which to compute the wrapped pdf
    :type thetas: float
    :return: Array of wrapped normal pdf values
    :rtype: ndarray
    """

    shifts = np.arange(-1000, 1001, 1)  # Make end one more than beginning so symmetry is preserved

    wn = [np.exp(-1 * (thetas - (mu - period * shift) * np.ones(len(thetas))) ** 2 / (2 * sigma ** 2)) for
                      shift in shifts]

    wn = 1 / (sigma * np.sqrt(period)) * np.sum(np.array(wn), axis=0)

    return wn


def IfiberRad(a0, a2, a4, phi, theta):

    """
    Compute the double cosine series solution to Mie scattering of cylindrical fibers with radian inputs

    :param a0: a0 parameter
    :type a0: float
    :param a2: a2 parameter
    :type a2: float
    :param a4: a4 parameter
    :type a4: float
    :param phi: preferred fiber direction in degrees
    :type phi: float
    :param theta: Values of theta at which to evaluate the cosine series. Values in radians
    :type theta: ndarray
    :return: Intensity values
    :rtype: ndarray
    """

    rv = a0 + a2 * np.cos(2 * (theta - phi)) + a4 * np.cos(4 * (theta - phi))

    return rv


def IdistWrappedNormal(a0, a2, a4, phi, thetas, dist):

    """
    Simulate signal intensity resulting from the convolution of a single fiber with a wrapped normal distribution

    :param a0: a0 parameter
    :type a0: float
    :param a2: a2 parameter
    :type a2: float
    :param a4: a4 parameter
    :type a4: float
    :param phi: preferred fiber direction in degrees
    :type phi: float
    :param thetas: Values of theta at which to evaluate the cosine series. Values in radians
    :type thetas: ndarray
    :param dist: Values of the wrapped normal pdf at the theta values
    :type dist: ndarray
    :return: Distribution intensity values
    :rtype: ndarray
    """

    fiber = IfiberRad(a0, a2, a4, phi, thetas)

    delta = thetas[1] - thetas[0]

    return fftconvolve(dist, fiber, 'same') * delta


def minimandWrappedNormal(a0, a2, a4, phi, thetas, dist, data):

    """
    Minimand for computing the sum of squared differences between a fit wrapped normal intensity signal and measured
    data

    :param a0: a0 parameter
    :type a0: float
    :param a2: a2 parameter
    :type a2: float
    :param a4: a4 parameter
    :type a4: float
    :param phi: preferred fiber direction in degrees
    :type phi: float
    :param thetas: Values of theta at which to evaluate the cosine series. Values in radians
    :type thetas: ndarray
    :param dist: Values of the wrapped normal pdf at the theta values
    :type dist: ndarray
    :param data: Measured signal array
    :type data: ndarray
    :return: Sum of square difference between data and proposed fit distribution.
    :rtype: float
    """
    guess = IdistWrappedNormal(a0, a2, a4, phi, thetas, dist)

    guess_interp = interp1d(thetas, guess)

    feval = guess_interp(data[:, 0])

    diff = data[:, 1] - feval

    diffsq = np.square(diff, diff)

    ssd = np.sum(diffsq)

    return ssd


def minfunWrappedNormal(params, *args):

    """
    Wrapper function for wrapped normal fitting minimand. This function is to be passed to Scipy's minimization functions

    :param params: List of minimization parameters: a0, a2, a4
    :type params: list, ndarray
    :param args: Tuple containing necessary arguments to ellipse minimand: (Phi, Thetas, Distribution, Signal)
    :type args: tuple
    :return: Sum of square difference between data and proposed fit distribution.
    :rtype: float
    """

    return minimandWrappedNormal(params[0], params[1], params[2], *args)


def beta_dist(mean, sd):

    """
    Generate a beta distribution using a mean and standard deviation based on the [-pi/2, pi/2] interval

    :param mean: Mean of beta distribution. Radian value in interval [-pi/2, pi/2]
    :type mean: float
    :param sd: Standard deviation of beta distribution. Radian value in interval [0, pi/2]
    :type sd: float
    :return: Instance of the scipy beta random variable class
    :rtype: rv_continous.beta
    """

    # Map mean and standard deviation to [0, 1] range of the standard beta distribution
    mu = (mean + np.pi / 2) / np.pi
    sigma = sd / np.pi

    # Compute the shape parameters
    gamma = (mu ** 2 - mu ** 3 - sigma ** 2 * mu) / (sigma ** 2)
    delta = gamma * (1 - mu) / mu

    cv = beta(gamma, delta)

    return cv


def IfiberBetaInterval(a0, a2, a4, phi, theta):

    """
    Compute the double cosine series solution to Mie scattering of cylindrical fibers with radian inputs. Thetas are
    mapped to the [0, 1] interval to make them compatible.

    :param a0: a0 parameter
    :type a0: float
    :param a2: a2 parameter
    :type a2: float
    :param a4: a4 parameter
    :type a4: float
    :param phi: preferred fiber direction in degrees
    :type phi: float
    :param theta: Values of theta at which to evaluate the cosine series. Values in radians
    :type theta: ndarray
    :return: Intensity values
    :rtype: ndarray
    """

    thetaRad = theta * np.pi - np.pi / 2 * np.ones(len(theta))

    return IfiberRad(a0, a2, a4, phi, thetaRad)


def IdistBeta(a0, a2, a4, phi, thetas, dist):

    """
    Simulate signal intensity resulting from the convolution of a single fiber with a beta distribution

    :param a0: a0 parameter
    :type a0: float
    :param a2: a2 parameter
    :type a2: float
    :param a4: a4 parameter
    :type a4: float
    :param phi: preferred fiber direction in degrees
    :type phi: float
    :param thetas: Values of theta at which to evaluate the cosine series. Values in radians
    :type thetas: ndarray
    :param dist: Values of the wrapped normal pdf at the theta values
    :type dist: ndarray
    :return: Distribution intensity values
    :rtype: ndarray
    """

    Ifiber = IfiberBetaInterval(a0, a2, a4, phi, thetas)

    dist_y = dist.pdf(thetas)

    delta = thetas[1] - thetas[0]

    return fftconvolve(dist_y, Ifiber, 'same') * delta


def minimandBeta(a0, a2, a4, phi, thetas, dist, data):

    """
    Minimand for computing the sum of squared differences between a fit beta distribution intensity signal and measured
    data

    :param a0: a0 parameter
    :type a0: float
    :param a2: a2 parameter
    :type a2: float
    :param a4: a4 parameter
    :type a4: float
    :param phi: preferred fiber direction in degrees
    :type phi: float
    :param thetas: Values of theta at which to evaluate the cosine series. Values in radians
    :type thetas: ndarray
    :param dist: Values of the beta pdf at the theta values
    :type dist: ndarray
    :param data: Measured signal array
    :type data: ndarray
    :return: Sum of square difference between data and proposed fit distribution.
    :rtype: float
    """

    guess = IdistBeta(a0, a2, a4, phi, thetas, dist)

    guess_interp = interp1d(thetas, guess)

    # Scale the measured psfdi points to the [0, 1] interval
    scaled_data = (data[:, 0] + np.pi / 2 * np.ones(len(data))) / np.pi
    scaled_data = np.round(scaled_data, 3)  # This isn't the best approach, but okay for now

    feval = guess_interp(scaled_data)

    diff = data[:, 1] - feval

    diffsq = np.square(diff, diff)

    ssd = np.sum(diffsq)

    return ssd


def minfunBeta(params, *args):

    """
    Wrapper function for beta fitting minimand. This function is to be passed to Scipy's minimization functions

    :param params: List of minimization parameters: a0, a2, a4
    :type params: list, ndarray
    :param args: Tuple containing necessary arguments to ellipse minimand: (Phi, Thetas, Distribution, Signal)
    :type args: tuple
    :return: Sum of square difference between data and proposed fit distribution.
    :rtype: float
    """
    return minimandBeta(params[0], params[1], params[2], *args)


def cdf_gamma(gamma_data, theta_data, theta):

    """
    Compute the cumulative distribution function of a give orientation distribution function at a specified theta.

    :param gamma_data: Array of ODF data
    :type gamma_data: ndarray
    :param theta_data: Array of theta values
    :type theta_data: ndarray
    :param theta: Argument of cdf. Radian in interval [-pi/2, pi/2]
    :type theta: float
    :return: Value of cdf at specified theta
    :rtype: float
    """

    indices = theta_data <= theta

    gammas = gamma_data[indices]
    thetas = theta_data[indices]

    return trapz(gammas, thetas)


def IdistDiscrete(a0, a2, a4, phis, thetas):

    """
    Simulate signal intensity resulting from the convolution of a single fiber with a discretely sampled distribution

    :param a0: a0 parameter
    :type a0: float
    :param a2: a2 parameter
    :type a2: float
    :param a4: a4 parameter
    :type a4: float
    :param phis: Preferred directions sampled from distribution
    :type phis: ndarray
    :param thetas: Values of theta at which to evaluate the cosine series. Values in radians
    :type thetas: ndarray
    :return: Distribution intensity values
    :rtype: ndarray
    """

    rv = np.zeros((len(phis), len(thetas)))

    for row in range(len(phis)):
        rv[row, :] = IfiberRad(a0, a2, a4, phis[row], thetas)

    rv = np.sum(rv, axis=0) / len(phis)

    return rv


def minimandDiscrete(a0, a2, a4, phis, thetas, data):

    """
    Minimand for computing the sum of squared differences between a discretely sampled distribution intensity signal and measured
    data

    :param a0: a0 parameter
    :type a0: float
    :param a2: a2 parameter
    :type a2: float
    :param a4: a4 parameter
    :type a4: float
    :param phis: Preferred directions sampled from distribution
    :type phis: ndarray
    :param thetas: Values of theta at which to evaluate the cosine series. Values in radians
    :type thetas: ndarray
    :param data: Measured signal array
    :type data: ndarray
    :return: Sum of square difference between data and proposed fit distribution.
    :rtype: float
    """

    feval = IdistDiscrete(a0, a2, a4, phis, thetas)

    diff = data - feval

    diffsq = np.square(diff, diff)

    ssd = np.sum(diffsq)

    return ssd


def minfunDiscrete(params, *args):

    """
    Wrapper function for discrete distribution fitting minimand. This function is to be passed to Scipy's minimization functions

    :param params: List of minimization parameters: a0, a2, a4
    :type params: list, ndarray
    :param args: Tuple containing necessary arguments to ellipse minimand: (Phi, Thetas, Signal)
    :type args: tuple
    :return: Sum of square difference between data and proposed fit distribution.
    :rtype: float
    """

    return minimandDiscrete(params[0], params[1], params[2], *args)
