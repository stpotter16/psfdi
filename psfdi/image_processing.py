"""
.. module:: image_processing
    :platform: Unix, Windows
    :synopsis: Methods for manipulating and processing raw image data

.. moduleauthor:: Sam Potter <spotter1642@gmail.com>
"""

from . import np
from . import cv2


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
    Function for upscaling an image to a lower pixel resolution by using simple averaging to create new pixel values

    :param image: Image to be upscaled. Shape (rows x columns)
    :type image: ndarray
    :param new_rows: Number of rows in lower resolution image
    :type new_rows: int
    :param new_columns: Number of columns in lower resolution image
    :type new_columns: int
    :return: Upscaled image. Shape (new_rows x new columns)
    :rtype: ndarray
    """

    org_rows, org_columns = image.shape

    row_step = int(org_rows / new_rows)
    col_step = int(org_columns / new_columns)

    image_upscaled = np.zeros((new_rows, new_columns))

    for i in range(0, new_rows):
        for j in range(0, new_columns):
            subar = image[i * row_step: i * row_step + row_step, j * col_step: j * col_step + col_step]
            val = np.mean(subar)
            image_upscaled[i, j] = val

    return image_upscaled


def rotate(image, angle):

    """
    Function wraps opencv's methods for rotating an image and returning it with the same shape as the input. Linear
    interpolation is used

    :param image: Image data as numpy array. Shape (rows x columns)
    :type image: ndarray
    :param angle: Angle of rotation.
    :type angle: float
    :return: Rotated image as numpy array. Shape (rows x columns)
    :rtype: ndarray
    """

    sz = image.shape

    rot_mat = cv2.getRotationMatrix2D((sz[1] / 2, sz[0] / 2), angle, 1)

    image_rotated = cv2.warpAffine(image, rot_mat, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return image_rotated


def register(image, target_image, niters=5000, eps=1e-5):

    """
    This function registers image to a target image by finding the warp that minimizes the correlation coefficient
    between the warped image and the target.

    The function then applies this warp the the image.

    It is assumed that a euclidean model of motion describes the spatial relationship between the image and its target.

    Image and target must have the same shape and size

    :param image: Image to be registered. A numpy array of shape (rows x columns).
    :type image: ndarray
    :param target_image: Target image for registration. A numpy array of shape (rows x columns)
    :type target_image: ndarray
    :param niters: Optional. Max number of iterations for registration algorithm to run while trying to find the warp.
    Default 5000
    :type niters: int
    :param eps: Optional. Termination condition on correlation coefficient in warp finding minimizer. Default 1e-5
    :return: Tuple containing warped image and dictionary of warp data, i.e. correlation coefficient and warp matrix
    :rtype: tuple
    """

    #TODO Check image and target_image are same shape and size

    # Dict for holding warp info

    warp_dict = {}

    # Find warp
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    warp_mode = cv2.MOTION_EUCLIDEAN

    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, niters, eps)

    (cc, warp_matrix) = cv2.findTransformECC(target_image, image, warp_matrix, warp_mode, criteria, None, 5)

    # Wrap warp outputs into a dict
    warp_dict['cc'] = cc
    warp_dict['warp'] = warp_matrix

    # Warp image
    sz = target_image.shape

    registered_image = cv2.warpAffine(image, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return registered_image, warp_dict


def warp(image, warp_matrix):

    """
    Function for applying a registration warp to an image.

    Typically the registration warp is found with the function register.

    Warp assumes the output image size is the same as the input and uses linear interpolation

    :param image: Image to be warped. A numpy array of shape (rows x columns).
    :type image: ndarray
    :param warp_matrix: Matrix describing warp transformation. Shape (2, 3). Data type is 32 bit float
    :type ndarry
    :return: Warped image as a numpy array. Shape (rows x columns)
    :rtype: ndarray
    """

    # Warp
    sz = image.shape

    warped_image = cv2.warpAffine(image, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return warped_image
