"""
.. module:: fileIO
    :platform: Unix, Windows
    :synopsis: Methods for reading and writing to files (image data, etc)

.. moduleauthor:: Sam Potter <spotter1642@gmail.com>
"""

from . import np
from . import os
from . import cv2
from . import errno
from . import regex
from . import sio
from . import utilities
from . import cm


def read_mat_file(fname):

    """
    Read in MATLAB .mat file and return the dictionary of values

    :param fname: Name of input file including extension
    :type fname: str
    :return: Dictionary containing .mat file contents
    :rtype: dict
    """
    # Input checking
    # This check does not exactly conform to "Duck Typing", but I think it's useful because the exception thrown by
    # the Scipy IO module is a little vague and can be confusing.
    if not fname.endswith('.mat'):
        raise ValueError('Input file should have a .mat extension, e.g. my_matlab_data.mat')

    data_dict = sio.loadmat(fname)

    return data_dict


def write_tiff_stack(data_array, base_name, directory=None, color_map='viridis'):

    """
    Writes a 3 dimensional array of data to 16 bit tiff image stacks.

    Allows options for specifying input directory and color map
    :param data_array: Data array to write to image. Shape Number of Image Steps x Image X Size x Image Y Size.
    Assumed that input is not normalized.
    :type data_array: numpy array
    :param base_name: base name string of output file. Will be appended with image index number and .tiff extension
    :type base_name: str
    :param directory: Optional. Full path to write output files into. Default will be current working directory of
    script
    :type directory: str
    :param color_map: Optional. Matplotlib colormap to use for image write.
    :type color_map: str
    :return: None
    :rtype: None
    """
    # Input checking
    if len(data_array.shape) != 3:
        raise ValueError('Data array must be three dimensional!')

    # Check if directory needs to be changes
    if directory:
        # Save current directory for later use
        start = os.getcwd()

        # Change directory
        # Attempt to move into directory
        try:
            os.chdir(directory)
        except OSError as ex:
            if ex.errno != errno.ENOENT:
                raise
        # If directory doesn't exist, try to make it
        try:
            os.makedirs(directory)
            os.chdir(directory)
        except OSError:
            print('Unable to move into or create requested output directory')
            print(directory)
            raise

    # Loop through every image in array and write output image
    for i in range(len(data_array)):

        # Normalize input data to [0, 1]
        utilities.normalize_2d(data_array[i, :, :])

        # Get colormap
        colormap = cm.get_cmap(color_map)

        # Apply colormap to data
        im_data = colormap(data_array[i, :, :])

        # Scale to 16 bit range
        im_data = 65535 * im_data

        # Convert to unsigned 16 int type
        im_data = np.uint16(im_data)

        # Flip RGB order to BGR order so it's compatible with OpenCV
        im_data = im_data[..., ::-1]

        # Create file output name
        fname = base_name + '_' + str(i) + '.tiff'

        # Write file
        cv2.imwrite(fname, im_data)

    if directory:
        # Move directory back to start
        os.chdir(start)


def write_pSFDI_tiffs(data_dict, directory=None):

    """
    Writes all the pSFDI imaging data contained in the input dictionary as tiff stacks in their own unique directories

    Allows for the user specification of top level output directory.

    :param data_dict: Dictionary containing pSFDI data as arrays. Format: {name of pSFDI measure: 3D array of data}.
    Data is formatted as Imaging Step Number x Image Size X x Image Size Y
    :type data_dict: dict
    :param directory: Optional. Full path specification of top level output directory
    :type directory: str
    :return: None
    :rtype: None
    """
    # Check if directory needs to be changed
    if directory:
        # Save current directory for later use
        start = os.getcwd()

        # Change directory
        # Attempt to move into directory
        try:
            os.chdir(directory)
        except OSError as ex:
            if ex.errno != errno.ENOENT:
                raise
        # If directory doesn't exist, try to make it
        try:
            os.makedirs(directory)
            os.chdir(directory)
        except OSError:
            print('Unable to move into or create requested directory')
            print(directory)
            raise

    # Loop over each element in the pSFDI data dictionary
    for i in iter(data_dict):
        # This could be a one-liner, but "readability counts" and so does clarity
        # Extract the data array
        data_array = data_dict[i]
        # Extract the file name based on key name
        base_name = i[3:]  # Chopping off first part of key name, e.g. 'ag_a0_DC' -> 'a0_DC'. BRITTLE
        # Set color map based on the key name
        if regex.findall('a0', base_name):
            colormap = 'gray'
        elif regex.findall('phi', base_name):
            colormap = 'hsv'
        else:
            colormap = 'viridis'
        # Call tiff stack writer with data array, name, directory (based on filename), and color map
        write_tiff_stack(data_array, base_name, base_name, color_map=colormap)

    if directory:
        os.chdir(start)


def read_psfdi_mask(data_path):

    '''
    This function reads and returns the data mask created during pSFDI processing in matlab

    :param data_path: Absolute path to data folder
    :type data_path: str
    :return: tuple of rows, columns of the mask (xrange, yrange)
    :rtype: tuple
    '''

    mask_path = os.path.join(data_path, 'cmask.mat')
    mask_dict = read_mat_file(mask_path)
    xrange = mask_dict['xrange']
    yrange = mask_dict['yrange']

    return xrange, yrange


def read_raw_psfdi(data_path, xrange, yrange, sfx_per, polar_res, polar_max):

    '''
    This function reads in the raw pSFDI image data and returns 3D arrays of the image sets.

    :param data_path: Absolute path to data folder
    :type data_path: str
    :param xrange: numpy array of row index corresponding to data mask
    :type xrange: ndarray
    :param yrange: numpy array of column index corresponding to  data mask
    :type yrange: ndarray
    :param sfx_per: Period of the spatial frequency images to load
    :type sfx_per: float
    :param polar_max: Max angle over which polarizer was rotated
    :type polar_max: int
    :param polar_res: Resolution of polarizer used to collect images
    :type polar_res: float
    :return: Tuple of of image data (dark images, planar images, I0, I120, I240)
    :rtype: tuple
    '''

    # Read images with no spatial frequency
    dark = read_no_sfx(data_path, xrange, yrange, polar_res, polar_max, 'dark')
    planar = read_no_sfx(data_path, xrange, yrange, polar_res, polar_max, 'planar')

    # Read images with spatial frequency

    return dark, planar


def read_no_sfx(data_path, xrange, yrange, polar_res, polar_max, imtype):

    '''
    This function reads and crops image data that does not have a spatial frequency pattern projected on to it

    :param data_path: Absolute path to data folder
    :type data_path: str
    :param xrange: numpy array of row index corresponding to data mask
    :type xrange: ndarray
    :param yrange: numpy array of column index corresponding to  data mask
    :type yrange: ndarray
    :param polar_res: Resolution of polarizer used to collect images
    :type polar_res: float
    :param polar_max: Max angle over which polarizer was rotated
    :type polar_max: int
    :param imtype: Type of non spatial frequency image to read. Options are dark, planar.
    :type imtype: str
    :return: Array of cropped image data
    :rtype: ndarray
    '''

    # Set polar angles
    polar_angles = np.arange(0, polar_max, polar_res)

    # Read in files
    images = [cv2.imread(os.path.join(data_path, imtype+'_angle_' + str(angle) + '.tiff'), -1)
              for angle in polar_angles]
    # Convert to numpy array
    images = np.array(images)

    # Crop
    # Do this in steps to avoid broadcasting issues. EVALUATE LATER
    images = images[:, xrange.tolist()[0], :]
    images = images[:, :, yrange.tolist()[0]]

    return images


def read_sfx(data_path, xrange, yrange, polar_res, polar_max, sfx_per, phase_shift):

    '''
    This function reads and crops image data that does have a spatial frequency pattern projected on to it

    :param data_path: Absolute path to data folder
    :type data_path: str
    :param xrange: numpy array of row index corresponding to data mask
    :type xrange: ndarray
    :param yrange: numpy array of column index corresponding to  data mask
    :type yrange: ndarray
    :param polar_res: Resolution of polarizer used to collect images
    :type polar_res: float
    :param polar_max: Max angle over which polarizer was rotated
    :type polar_max: int
    :param sfx_per: period of spatial frequency pattern projected on the image in pixels
    :type sfx_per: float
    :param phase_shift: phase shift to read
    :type phase_shift: int
    :return: Array of cropped image data
    :rtype: ndarray
    '''

    # Set polar angles
    polar_angles = np.arange(0, polar_max, polar_res)

    # Read in files
    images = [cv2.imread(os.path.join(data_path, 'V_sf_' + str(sfx_per) + '_pha_' + str(phase_shift) + '_angle_'
                                      + str(angle) + '.tiff'), -1) for angle in polar_angles]
    # Convert to numpy array
    images = np.array(images)

    # Crop
    # Do this in steps to avoid broadcasting issues. EVALUATE LATER
    images = images[:, xrange.tolist()[0], :]
    images = images[:, :, yrange.tolist()[0]]

    return images
