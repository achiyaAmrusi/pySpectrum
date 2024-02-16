import numpy as np
from pyspectrum.spectrum import Spectrum
from pyspectrum.peak_identification.peaks_fit import GaussianWithBGFitting
from pyspectrum.peak_identification.zero_area_functions import gaussian_2_dev
from uncertainties import ufloat

EPSILON = 1e-4


class Convolution:
    """
    tool to convolve a function (domain and range) with a kernel of zero area function.
    TODO: change the convolution method to do the convolution. in order to save time i can calculate the kernel
          on twice the domain. than there is no need to call kernel in each step of the loop. Instead, I can each
          step of the loop just cut the relevant kernel part
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


class FindPeaks:
    """
        Find peaks is a tool for general peak finding using the convolution method.
        the purpose of the work is to be a tool for spectroscopy but this can be generalized.
        FindPeaks take a Spectrum and using the calibration to find the gamma and xray peaks in the spectrum
        The code uses method represents in the paper[][].

        Attributes:
        __________
        spectrum: Spectrum
         The spectrum that FindPeaks search and return peaks in.
        convolution: Convolution
         The convolution of the spectrum with the kernel according to []
         This kernel_convolution is calculated only once due to the time intensive cost of calculating it
        fitting_type: 'str'
        the type of the fitting method. for now there is only 'HPGe_spectroscopy'.
        TODO: add more options for the fitting method.
         I can add simple gaussian with a simple constant background, a alpha spectrum and others?

        Methods:
        ________

    - `__init__(self, xarray.DataArray)`:
      Constructor method to initialize a FindPeaks instance.

    - `find_all_peaks(self, stat_error=0.05, fwhm_tol_min=0.5)`:
      find peaks in the gamma spectrum given the acceptable statistical error
      and the fwhm tolerance from the given fwhm function in spectrum(Spectrum.fwhm_calibration)

    - `find_all_peaks(self, stat_error=0.05, fwhm_tol_min=0.5)`:
      find peaks in the gamma spectrum given the acceptable statistical error
      and the fwhm tolerance from the given fwhm function in spectrum(Spectrum.fwhm_calibration)

    - `plot_all_peaks(self, stat_error=0.05, fwhm_tol_min=0.5)`:
      find the peaks using find_all_peaks and plot them

    - `peak(self, peak_center:list, channel_mode=False)`:
      for a given peak_center list the function returns all the Peak object of the peaks around the peak_centers
      The domain of the peak is calculated using convolution method
    """

    def __init__(self, spectrum: Spectrum, convolution: Convolution, fitting_type='HPGe_spectroscopy'):
        """
        Constructor method to initialize a FindPeaks instance
        Parameters
        ----------
        spectrum: Spectrum
         The spectrum that FindPeaks search and return peaks in.
        convolution: Convolution
         The convolution of the spectrum with the chosen kernel.
         This kernel_convolution is calculated only once due to the time intensive cost of calculating it
        fitting_type: 'str'
        the type of the fitting method. for now there is only
        'HPGe_spectroscopy'
        Returns
        -------
"""
        self.spectrum = spectrum
        self.convolution = convolution
        _, _, self.conv_n_sigma_spectrum = convolution.convolution(spectrum.channels, spectrum.counts)
        if fitting_type == 'HPGe_spectroscopy':
            self.fitting_method = GaussianWithBGFitting(False)
        else:
            raise f'{fitting_type} fitting method dose not exist'

    def peaks_domain(self, value_in_domain):
        """
        find signal peak domain on which the channel is in using the threshold of FindPeaks
        Parameters
        ----------
        value_in_domain: float
        value in the domain which is inside a peak.
        Returns
        -------
        peak channels domain: tuple
            domain border from the left, domain border from the right

        References
        ----------
        [1]Phillips, Gary W., and Keith W. Marlow.
         "Automatic analysis of gamma-ray spectra from germanium detectors."
          Nuclear Instruments and Methods 137.3 (1976): 525-536.

        """
        spectral_n_sigma = self.conv_n_sigma_spectrum
        n_sigma_threshold = self.convolution.n_sigma_threshold
        # calculate the channel of the energy value
        channel = np.abs(self.spectrum.energy_calibration(self.spectrum.channels) - value_in_domain).argmin()
        ch_fwhm = self.spectrum.fwhm_calibration(
            self.spectrum.energy_calibration(channel)) / self.spectrum.energy_calibration[1]
        if ch_fwhm < 2:
            ch_fwhm = 2

        # find peak left edge
        mean_n_sigma_fwhm = n_sigma_threshold
        channel_l = channel + 1
        while spectral_n_sigma[channel_l] > n_sigma_threshold / 2 or mean_n_sigma_fwhm >= n_sigma_threshold / 2:
            channel_l = channel_l - 1
            mean_n_sigma_fwhm = (spectral_n_sigma[channel_l - round(ch_fwhm):channel_l]).mean()
        # find peak right edge
        mean_n_sigma_fwhm = n_sigma_threshold
        channel_r = channel - 1
        while spectral_n_sigma[channel_r] > n_sigma_threshold / 2 or mean_n_sigma_fwhm >= n_sigma_threshold / 2:
            channel_r = channel_r + 1
            mean_n_sigma_fwhm = (spectral_n_sigma[channel_r:channel_r + round(ch_fwhm)]).mean()
        return channel_l - 1, channel_r + 1

    def find_all_peaks(self):
        """
        The function finds peaks automatically
        function scan the spectral_n_sigma vector which is kernel_convolution/kernel_convolution_std
        and search for when the value is larger than 4, if the value is above 4, it calls peak_domain.
        the peaks are than fitted and returned as Peaks and Peaks properties

        Parameters
        ----------

        Returns
        -------
        - tuple
        peaks_properties: list
        a list of the properties of each peak found.
        peak properties  = (peak_domain:tuple, properties of fit curve in xarray)

        References
        ----------
        [1]Phillips, Gary W., and Keith W. Marlow.
         "Automatic analysis of gamma-ray spectra from germanium detectors."
          Nuclear Instruments and Methods 137.3 (1976): 525-536.
        [2]Likar, Andrej, and Tim Vidmar.
        "A peak-search method based on spectrum convolution."
        Journal of Physics D: Applied Physics 36.15 (2003): 1903.
        """

        peaks_domain = self.find_all_peaks_domain()
        spectrum = self.spectrum.counts
        peaks_properties = []
        for peak_domain in peaks_domain:
            # fwhm in the peak
            middle_channel = round((peak_domain[0] + peak_domain[1]) / 2)
            ch_fwhm = self.spectrum.fwhm_calibration(
                self.spectrum.energy_calibration(middle_channel)) / self.spectrum.energy_calibration[1]

            # the background levels from left and right
            peak_bg_l = ufloat(spectrum[peak_domain[0] - round(ch_fwhm / 2):peak_domain[0]].mean(),
                               spectrum[peak_domain[0] - round(ch_fwhm / 2):peak_domain[0]].std())
            peak_bg_r = ufloat(spectrum[peak_domain[1]:peak_domain[1] + round(ch_fwhm / 2)].mean(),
                               spectrum[peak_domain[1]:peak_domain[1] + round(ch_fwhm / 2)].std())
            # the peak
            peak = self.spectrum.xr_spectrum().sel(energy=slice(self.spectrum.energy_calibration(peak_domain[0]),
                                                                self.spectrum.energy_calibration(peak_domain[1])))
            # fit the peak using the fitting method
            peak_background_data = [peak_bg_l - peak_bg_r, peak_bg_r]
            fit_properties, num_of_peaks = self.fitting_method.peak_fit(peak, peak_background_data)
            # save all the peak found in the domain
            for fit in fit_properties:
                peaks_properties.append((peak_domain, fit))
        return peaks_properties

    def plot_all_peaks(self):
        """plot the peaks found in find_peaks
        Parameters
        ----------

        Returns
        -------

   """
        peaks_properties = self.find_all_peaks()
        self.spectrum.xr_spectrum().plot()
        for peak in peaks_properties:
            energy_domain = self.spectrum.energy_calibration(np.arange(peak[0][0], peak[0][1], 1))
            self.fitting_method.plot_fit(energy_domain, peak[1])



