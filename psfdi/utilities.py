"""
.. module:: utilities`
    :platform: Unix, Windows
    :synopsis: Utility and helper functions

.. moduleauthor:: Sam Potter <spotter1642@gmail.com>
"""

from . import np


def normalize_2d(data_array):

    """
    Normalize a 2D data array to the range [0, 1]. NOTE, FUNCTION OPERATES ON ARRAY IN PLACE AND DOES NOT COPY AND
    RETURN. Take appropriate caution

    :param data_array: Input 2D data array.
    :type data_array: numpy array
    :return: None
    :rtype: None
    """
    # Check inputs (array must be two dimensional)
    if data_array.ndim != 2:
        raise ValueError('Input data must be 2D!')

    # Get max and min values
    data_max = np.max(data_array)
    data_min = np.min(data_array)

    # Scale values to range [0, 1]
    data_array = data_array - data_min * np.ones(data_array.shape)  # Subtract off min value
    data_array *= 1/(data_max - data_min)  # Multiply by scaling term


def normalize_1d(data_array):
    
    """
    Normalize a 1D data array to range [0, 1].
    
    :param data_array: Input 1d data
    :type data_array: ndarray
    :return: None
    :rtype: None
    """
    # Sanitize input
    if data_array.ndim != 1:
        raise ValueError('Input data must be 1D!')
    
    data_max = np.max(data_array)
    data_min = np.min(data_array)
    
    data_array = data_array - data_min * np.ones(data_array.shape)
    data_array *= 1/(data_max - data_min)
    
    return data_array
