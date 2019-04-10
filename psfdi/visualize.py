"""
.. module:: visualization
    :platform: Unix, Windows
    :synopsis: Module contains functions visualizing pSFDI, SALS, ODF, and fiber scattering data

.. moduleauthor:: Sam Potter <spotter1642@gmail.com>
"""

from . import np
from . import odf
from . import plt
from . import utilities


def Ifiber_interactive(a0, a2, a4, phi, theta_min, theta_max, theta_numpts, normalize=True):

    '''
    Function for interactive visualization of single fiber Mie scattering

    :param a0: a0 parameter
    :type a0: float
    :param a2: a2 parameter
    :type a2: float
    :param a4: a4 parameter
    :type a4: float
    :param phi: preferred fiber direction in degrees
    :type phi: float
    :param theta_min: minimum value of the theta span
    :type theta_min: float
    :param theta_max: maximum value of the theta span
    :type theta_max: float
    :param theta_numpts: number of points in the theta span
    :type theta_numpts: int
    :param normalize: Bool switch on whether or not to normalize results
    :type normalize: bool
    :return:
    '''

    theta = np.linspace(theta_min, theta_max, theta_numpts)

    vals = odf.Ifiber(a0, a2, a4, phi, theta)

    if normalize:
        vals = utilities.normalize_1d(vals)

    fig = plt.figure(figsize=(10, 10))
    plt.plot(theta, vals, color='g', label='Intensity');
    plt.legend(prop={'size': 18});
    plt.xlabel(r'$\theta$', fontsize=18);
    plt.ylabel('Normalized Intensity (a.u.)', fontsize=18);
    plt.title('Normalized Intensity Curves of Single Fiber under Cylindrical Scattering', fontsize=18);
    plt.autoscale(enable=True, axis='x', tight=True)

    print('a2/a4: {}'.format(a2 / a4))


def Idistribution_compare_interactive(a0, a2, a4, phi0, theta_min, theta_max, theta_numpts, splay, nsamples,
                                      distribution='uniform'):

    '''
    Function for interactive visualization of single fiber Mie scattering

    :param a0: a0 parameter
    :type a0: float
    :param a2: a2 parameter
    :type a2: float
    :param a4: a4 parameter
    :type a4: float
    :param phi0: preferred fiber direction in degrees
    :type phi0: float
    :param theta_min: minimum value of the theta span
    :type theta_min: float
    :param theta_max: maximum value of the theta span
    :type theta_max: float
    :param theta_numpts: number of points in the theta span
    :type theta_numpts: int
    :param splay: standard deviation of fiber direction distribution
    :type splay: float
    :param nsamples: number of samples to draw from the statistical distribution
    :type nsamples: int
    :param distribution: Optional. Specify type of distribution to draw samples from. Default is uniform
    :type distribution: str
    '''

    theta = np.linspace(theta_min, theta_max, theta_numpts)

    Idist = np.zeros((nsamples, len(theta)))
    if distribution == 'uniform':
        phis = np.random.uniform(-1 * splay, splay, nsamples)
    elif distribution == 'normal':
        phis = np.random.normal(0, splay, nsamples)
    for i in range(0, nsamples):
        phi = phis[i]
        vals = odf.Ifiber(a0, a2, a4, phi, theta)
        Idist[i, :] = vals

    Idist = np.sum(Idist, axis=0) / nsamples

    Ifibers = odf.Ifiber(a0, a2, a4, phi0, theta)

    fig = plt.figure(figsize=(10, 10))
    plt.plot(theta, Ifibers, color='g', label='Single Fiber Intensity');
    plt.plot(theta, Idist, color='r', label='Fiber Distribution Intensity');
    plt.legend(prop={'size': 14}, loc='best');
    plt.xlabel(r'$\theta$', fontsize=18);
    plt.ylabel('Intensity (a.u.)', fontsize=18);
    plt.title('Intensity Curves from Single Fiber and a Distribution of Fibers', fontsize=18);
    # plt.ylim(0, 1.25)
    plt.autoscale(enable=True, axis='x', tight=True)

    fig = plt.figure(figsize=(10, 10))
    plt.hist(phis)
    plt.title('Historgram of phi')
    plt.xlabel('phi')
