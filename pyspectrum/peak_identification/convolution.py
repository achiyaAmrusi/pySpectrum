import numpy as np
from pyspectrum.peak_identification.zero_area_functions import gaussian_2_dev


EPSILON = 1e-4


class Convolution:
    """
    Tool to convolve a function (domain and range) with a kernel of zero area function.
    TODO: change the convolution method to do the convolution. in order to save time i can calculate the kernel
          on twice the domain. than there is no need to call kernel in each step of the loop. Instead, I can each
          step of the loop just cut the relevant kernel part
    Parameters
     ----------
     width: Callable
         for a given bin returns the approximated width of the peaks
     zero_area_function: Callable
         a function that takes a domain of bins, a bin (in the domain) which is the center, and the threshold for
     n_sigma such that the if the value of the convolution(bin) > n_sigma there is a peak in bin
     n_sigma_threshold: float
         n_sigma such that the if the value of the convolution(bin) > n_sigma there is a peak in bin

    Attributes:
    __________
    width: Callable
         for a given bin returns the approximated width of the peaks
     zero_area_function: Callable
         a function that takes a domain of bins, a bin (in the domain) which is the center, and the threshold for
     n_sigma such that the if the value of the convolution(bin) > n_sigma there is a peak in bin
     n_sigma_threshold: float
         n_sigma such that the if the value of the convolution(bin) > n_sigma there is a peak in bin
     Methods:
    ________
     `__init__(self, xarray.DataArray)`
       Constructor method to initialize a Convolution instance.
     `kernel(self, domain: np.array, center: float)`
         Generate the kernel function around the given center value(channel)
         the kernel width is defined in spectrum.fwhm_calibration.
         The kernel function here is the second derivative of gaussian
     convolution(self, domain: np.array, function_range: np.array)
         calculate the function convolution with the kernel - zero area function
    """

    def __init__(self, width, zero_area_function=gaussian_2_dev, n_sigma_threshold=4):
        """
        Constructor method to initialize a Convolution instance
        Parameters
        ----------
        width: Callable
            for a given bin returns the approximated width of the peaks
        zero_area_function: Callable
            a function that takes a domain of bins, a bin (in the domain) which is the center, and the threshold for
        n_sigma such that the if the value of the convolution(bin) > n_sigma there is a peak in bin
        n_sigma_threshold: float
            n_sigma such that the if the value of the convolution(bin) > n_sigma there is a peak in bin

        """
        self.width = width
        self.zero_area_function = zero_area_function
        self.n_sigma_threshold = n_sigma_threshold

    def kernel(self, domain: np.array, center: float):
        """
        Generate the kernel function around the given center value(channel)
        the kernel width is defined in spectrum.fwhm_calibration.
        The kernel function here is the second derivative of gaussian
        Parameters
        ----------
        domain: np.array
            the channel of the spectrum (or other function)
        center: float
            the center of the kernel function
        Returns
        -------
        numpy array
            The zero area function over the given domain
        """
        return self.zero_area_function(domain, center, self.width(center))

    def convolution(self, domain: np.array, function_range: np.array):
        """
        calculate the function convolution with the kernel - zero area function

        Parameters
        ----------
        domain: np.array
            the channels/energy/domain of the function
        function_range: float
            the range of the function to be convoluted (counts of the spectrum)
        Returns
        -------
        tuple convolution, n_sigma
            convolution : The zero area function over the given domain
            n_sigma: the std of the convolution (in the paper)
        """
        conv = []
        conv_variance = []
        for i, x in enumerate(domain):
            kern_vector = self.kernel(domain, x)
            kern_vector_variance = kern_vector ** 2
            conv.append(np.dot(kern_vector, function_range))
            conv_variance.append(np.dot(kern_vector_variance, function_range))
        conv_n_sigma = np.array([conv[i] / conv_variance[i] ** 0.5 if conv_variance[i] != 0
                                 else 0 for i in range(len(conv))])
        conv = np.array(conv)
        conv_std = np.array(conv_variance) ** 0.5
        return conv, conv_std, abs(conv_n_sigma)
