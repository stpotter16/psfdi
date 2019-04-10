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
from . import sem
from . import make_axes_locatable
from . import patches


def Ifiber_interactive(a0, a2, a4, phi, theta_min, theta_max, theta_numpts, normalize=True):

    """
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
    """

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

    """
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
    """

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


def compare_raw_interactive(row, col, psfdi_data, name, SALS_data):

    """
    This function is used to visualize and compare pSFDI intensity data and pSFDI ODF

    :param row: Row number of comparison pixel in SALS data
    :type row: int
    :param col: Column number of comparison pixel in SALS data
    :type col: int
    :param psfdi_data: Multidimensional array of raw pSFDI intensity data.
    Shape (Number of polarizer steps x rows x columns)
    :type psfdi_data: ndarray
    :param name: String used to make plot titles. Should be either 'DC' or 'AC' depending on data imaged
    :type name: str
    :param SALS_data: Dictionary of SALS data used for comparison.
    :type SALS_data: dict
    """

    # Get SALS data
    PD_2d = SALS_data['PD']
    SD_2d = SALS_data['SD']
    SALS_odf = SALS_data['odf']

    # define ydim as number of rows
    ydim = PD_2d.shape[0]

    # Define row step and column step. Note psfdi_data is three dimensional
    row_step = int(psfdi_data.shape[1] / PD_2d.shape[0])
    col_step = int(psfdi_data.shape[2] / PD_2d.shape[1])

    # Get the correct rectangle start indices
    psfdi_row = row * row_step
    psfdi_col = col * col_step

    # Plot images
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 20))
    im0 = ax0.imshow(psfdi_data[0, :, :], cmap='gray')
    rect = patches.Rectangle((psfdi_col, psfdi_row), row_step, col_step, edgecolor='r', facecolor='r')
    ax0.add_patch(rect)
    ax0.set_title('pSFDI {} data - Pixel Location in Red'.format(name))

    im1 = ax1.imshow(PD_2d, cmap='hsv')
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes('right', size='5%', pad=0.05)
    colorlimits = (0, 180);
    im1.set_clim(colorlimits)
    fig.colorbar(im1, cax=cax1)
    rect = patches.Rectangle((col, row), 1, 1, edgecolor='k', facecolor='k')
    ax1.add_patch(rect)
    ax1.set_title('SALS PD - Pixel Location in Black');

    im2 = ax2.imshow(SD_2d, cmap='jet')
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes('right', size='5%', pad=0.05)
    colorlimits = (45, 55);
    im2.set_clim(colorlimits)
    fig.colorbar(im2, cax=cax2)
    rect = patches.Rectangle((col, row), 1, 1, edgecolor='k', facecolor='k')
    ax2.add_patch(rect)
    ax2.set_title('SALS SD - Pixel Location in Black');

    theta = np.linspace(0, 360, 360)

    # Plot pSFDI intensity of single pixel
    idist = psfdi_data[:, psfdi_row, psfdi_col]
    idist = np.append(idist, idist)
    psfdi_theta = np.linspace(0, 360, len(idist))
    gamma = SALS_odf[col * ydim + (ydim - col), :]

    # Plot average pSFDI intensity within SALS beam ROI
    sub_psfdi = np.zeros((row_step * col_step, len(psfdi_theta)))
    sub_row = 0
    for row in range(psfdi_row, psfdi_row + row_step):
        for col in range(psfdi_col, psfdi_col + col_step):
            temp = psfdi_data[:, row, col]
            temp = np.append(temp, temp)
            sub_psfdi[sub_row, :] = temp
            sub_row += 1

    sub_psfdi_mean = np.mean(sub_psfdi, axis=0)
    sub_psfdi_sem = sem(sub_psfdi, axis=0)

    # Plot Polar
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 15), subplot_kw=dict(projection='polar'))
    ax0.plot(np.deg2rad(psfdi_theta), sub_psfdi_mean, linestyle='--', marker='o', color='g',
             label='Mean Fiber Intensity');
    ax0.set_title('Mean pSFDI Raw Fiber Distribution Intensity In SALS Beam ROI - {} Data'.format(name));
    ax0.set_ylim([12100, 12250])

    ax1.plot(np.deg2rad(theta), gamma, color='r', label='SALS ODF');
    ax1.set_ylim([0, 0.25])
    ax1.set_title('SALS ODF')
    fig.suptitle('Polar Plots of Intensity and SALS ODF')

    # Plot Cartesian
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(15, 15))
    ax0.plot(psfdi_theta, sub_psfdi_mean, linestyle='--', marker='o', color='g', label='Mean Fiber Intensity');
    ax0.fill_between(psfdi_theta, sub_psfdi_mean + sub_psfdi_sem, sub_psfdi_mean - sub_psfdi_sem, color='gray',
                     alpha=0.2);
    ax0.set_xlabel('Theta')
    ax0.set_ylabel('Intensity [a.u.]');
    ax0.autoscale(enable=True, axis='x', tight=True)
    ax0.legend()
    ax0.set_title('Mean p/m pSFDI Raw Fiber Distribution Intensity In SALS Beam ROI - {} Data'.format(name));

    ax1.plot(theta, gamma, color='r', label='SALS ODF');
    ax1.legend();
    ax1.set_ylabel('Gamma(theta)')
    ax1.set_xlabel('Theta');
    ax1.autoscale(enable=True, axis='x', tight=True);
    ax1.set_title('SALS ODF')
    fig.suptitle('Cartesian Plots of Intensity and SALS ODF');
