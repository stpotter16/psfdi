"""
.. module:: image_processing
    :platform: Unix, Windows
    :synopsis: Methods for manipulating and processing raw image data

.. moduleauthor:: Sam Potter <spotter1642@gmail.com>
"""

from . import np


def demodulate(I0, I120, I240, polar_res, polar_max):

    """
    Function demodulates phase shifted spatial frequency images and returns the the DC and AC images

    :param I0: Array of spatial frequency images with 0 deg phase shift. Shape (Num images, image row, image columns)
    :type I0: ndarray
    :param I120: Array of spatial frequency images with 120 deg phase shift. Shape (Num images, image row,
    image columns)
    :type I120: ndarray
    :param I240: Array of spatial frequency images with 240 deg phase shift. Shape (Num images, image row,
    image columns)
    :type I240: ndarray
    :param polar_res: Resolution of polarizer used to collect images
    :type polar_res: float
    :param polar_max: Max angle over which polarizer was rotated
    :type polar_max: int
    :return: Tuple containing DC and AC images (DC, AC). Shape of DC and AC same as the phase shift image arrays
    :rtype: tuple
    """

    # DC images
    IDC = [(I0[angle, :, :] + I120[angle, :, :] + I240[angle, :, :]) / 3 for angle in range(int(polar_max / polar_res))]
    IDC = np.array(IDC)

    # AC images
    IAC = [np.sqrt(2*(np.square(I0[angle, :, :] - I120[angle, :, :]) + np.square(I120[angle, :, :] - I240[angle, :, :])
                      + np.square(I240[angle, :, :] - I0[angle, :, :])) / 3)
           for angle in range(int(polar_max / polar_res))]
    IAC = np.array(IAC)

    return IDC, IAC


def upscale(image, new_rows, new_columns):

    """

    :param image:
    :param new_rows:
    :param new_columns:
    :return:
    """
