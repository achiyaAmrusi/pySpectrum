import numpy as np
from pyspectrum.peak_identification.zero_area_functions import gaussian_2_dev


class Convolution:
    """
    Tool to convolve a function (domain and range) with a zero area function kernel which has a changing width.
    Parameters:
    ----------
     width: Callable
     for a given bin, width returns the approximated width of the peaks
     zero_area_function: Callable
     a callable function of a zero area function which means that the integral over the domain gives 0
    Attributes:
    ----------
    width: Callable
         for a given bin returns the approximated width of the peaks
     zero_area_function: Callable
     a callable function of a zero area function which means that the integral over the domain gives 0
    The function takes a domain, center and width.

     Methods:
     -------
     `__init__(self, width, zero_area_function=gaussian_2_dev)`
       Constructor method to initialize a Convolution instance.
     `kernel(self, domain: np.ndarray, center: float)`
         Generate the kernel function around the given center value(channel)
         the kernel width is defined in spectrum.fwhm_calibration.
         The kernel function here is the second derivative of gaussian
     convolution(self, domain: np.ndarray, function_range: np.ndarray)
         calculate the function convolution with the kernel - zero area function
    """

    def __init__(self, width, zero_area_function=gaussian_2_dev):
        """
        Constructor method to initialize a Convolution instance

        Parameters
        ----------
        width: Callable
            for a given bin returns the approximated width of the peaks
        zero_area_function: Callable
        a zero area function for the convolution
        """
        self.width = width
        self.zero_area_function = zero_area_function

    def kernel(self, domain: np.array, center: float):
        """
        Generate the kernel function around the given center value(channel)
        the kernel width is defined in spectrum.fwhm_calibration.
        The kernel function here is the second derivative of gaussian
        Parameters
        ----------
        domain: np.ndarray
            the channel of the spectrum (or other function)
        center: float
            the center of the kernel function
        Returns
        -------
        numpy array
            The zero area function over the given domain
        """
        return self.zero_area_function(domain, center, self.width(center))

    def convolution(self, domain: np.ndarray, function_range: np.ndarray):
        """
        calculate the function convolution with the kernel - zero area function
        Parameters
        ----------
        domain: np.ndarray
            the channels/energy/domain of the function
        function_range: float
            the mathematical range of the function to be convoluted (counts of the spectrum)
        Returns
        -------
        tuple convolution, n_sigma
            convolution : The zero area function over the given domain
            n_sigma: the std of the convolution (in the paper)
        """
        conv = np.zeros_like(domain)
        conv_variance = np.zeros_like(domain)
        for i, x in enumerate(domain):
            kern_vector = self.kernel(domain, x)
            kern_vector_variance = kern_vector ** 2
            conv[i] = np.dot(kern_vector, function_range)
            conv_variance[i] = np.dot(kern_vector_variance, function_range)
        conv_n_sigma = np.array([conv[i] / conv_variance[i] ** 0.5 if conv_variance[i] != 0
                                 else 0 for i in range(len(conv))])
        conv = conv
        conv_std = conv_variance ** 0.5
        return conv, conv_std, abs(conv_n_sigma)
