import numpy as np
from pyspectrum.peak_identification.convolution import Convolution
from pyspectrum.peak.peak import Peak
from uncertainties import ufloat


class FindPeaks:
    """
    Find peaks is a tool for general peak finding using the convolution method.
    the purpose of the work is to be a tool for gamma spectroscopy but this can be generalized for other peaks.
    FindPeaks take a Spectrum and using the calibration to find the gamma and xray peaks in the spectrum
    The code uses method represents in the papers[1,2].

    Attributes:
    ----------
    spectrum: Spectrum
     The spectrum that FindPeaks search and return peaks in.
    convolution: Convolution
     The convolution of the spectrum with the kernel according to []
     This kernel_convolution is calculated only once due to the time intensive cost of calculating it
    fitting_type: 'str'
    the type of the fitting method. for now there is only 'HPGe_spectroscopy'.

    Methods:
    -------
    `__init__(self, xarray.DataArray)`:
      Constructor method to initialize a FindPeaks instance.

    find_all_peaks(self, stat_error=0.05, fwhm_tol_min=0.5)`:
      find peaks in the gamma spectrum given the acceptable statistical error
      and the fwhm tolerance from the given fwhm function in spectrum(Spectrum.fwhm_calibration)

    `find_all_peaks(self, stat_error=0.05, fwhm_tol_min=0.5)`:
      find peaks in the gamma spectrum given the acceptable statistical error
      and the fwhm tolerance from the given fwhm function in spectrum(Spectrum.fwhm_calibration)

    `plot_all_peaks(self, stat_error=0.05, fwhm_tol_min=0.5)`:
      find the peaks using find_all_peaks and plot them

    `peak(self, peak_center:list, channel_mode=False)`:
      for a given peak_center list the function returns all the Peak object of the peaks around the peak_centers
      The domain of the peak is calculated using convolution method

    References
    ----------
    [1]Phillips, Gary W., and Keith W. Marlow.
     "Automatic analysis of gamma-ray spectra from germanium detectors."
      Nuclear Instruments and Methods 137.3 (1976): 525-536.
    [2]Likar, Andrej, and Tim Vidmar.
    "A peak-search method based on spectrum convolution."
    Journal of Physics D: Applied Physics 36.15 (2003): 1903.
    """

    def __init__(self, spectrum, convolution: Convolution, n_sigma_threshold=4):
        """
        Constructor method to initialize a FindPeaks instance

        Parameters
        ----------
        spectrum: Spectrum
         The spectrum that FindPeaks search and return peaks in.
        convolution: Convolution
         The convolution of the spectrum with the chosen kernel.
         This kernel_convolution is calculated only once due to the time intensive cost of calculating it
        n_sigma_threshold: float
            n_sigma such that the if the value of the convolution(bin) > n_sigma there is a peak in bin
        """
        self.spectrum = spectrum
        self.convolution = convolution
        self.n_sigma_threshold = n_sigma_threshold
        _, _, self.conv_n_sigma_spectrum = convolution.convolution(spectrum.energy_calibration(spectrum.channels), spectrum.counts)

    def peaks_domain(self, value_in_domain):
        """
        find signal peak domain on which the channel is in using the threshold of FindPeaks.
        Parameters
        ----------
        value_in_domain: float
        value in the domain which is inside a peak.

        Returns
        -------
        peak channels domain: tuple
            domain border from the left, domain border from the right
        """
        spectral_n_sigma = self.conv_n_sigma_spectrum
        n_sigma_threshold = self.n_sigma_threshold
        # calculate the channel of the energy value
        channel = np.abs(self.spectrum.energy_calibration(self.spectrum.channels) - value_in_domain).argmin()
        ch_fwhm = self.spectrum.fwhm_calibration(
            self.spectrum.energy_calibration(channel)) / self.spectrum.energy_calibration[1]
        if ch_fwhm < 2:
            ch_fwhm = 2

        # find peak left edge
        mean_n_sigma_fwhm = n_sigma_threshold
        channel_l = channel + 1
        while (spectral_n_sigma[channel_l] > n_sigma_threshold / 2 or mean_n_sigma_fwhm >= n_sigma_threshold / 2)\
                and channel_l > (self.spectrum.channels[1]):
            channel_l = channel_l - 1
            half_fwhm_distance = channel_l - round(ch_fwhm / 2)
            if half_fwhm_distance <= self.spectrum.channels[0]:
                half_fwhm_distance = self.spectrum.channels[0]
            mean_n_sigma_fwhm = (spectral_n_sigma[half_fwhm_distance:channel_l]).mean()

        # find peak right edge
        mean_n_sigma_fwhm = n_sigma_threshold
        channel_r = channel - 1
        while (spectral_n_sigma[channel_r] > n_sigma_threshold / 2 or mean_n_sigma_fwhm >= n_sigma_threshold / 2)\
                and channel_r < (self.spectrum.channels[-2]):
            channel_r = channel_r + 1
            half_fwhm_distance = channel_r + round(ch_fwhm / 2)
            if half_fwhm_distance >= self.spectrum.channels[-1]:
                half_fwhm_distance = self.spectrum.channels[-1]
            mean_n_sigma_fwhm = (spectral_n_sigma[channel_r:half_fwhm_distance]).mean()
        return channel_l - 1, channel_r + 1

    def find_domains_above_snr(self):
        """
        The function finds all the peaks domain  which are above the snr.
        The function scan the spectral_n_sigma vector which is kernel_convolution/kernel_convolution_std
        and search for when the value is larger than n_sigma_threshold, if the value is above n_sigma_threshold,
        it checks it finds the peak domain using self.peaks_domain
        Warning, function does not distinguish if the peak is a valid peak, only if it is above noise.
        Returns
        -------
        - tuple
        peaks_properties: list
        a list of the properties of each peak found.
        peak properties  = (peak_domain:tuple, properties of fit curve in xarray)
        """

        domains = []
        ind = 1
        channel = self.spectrum.channels[1]
        while ind < (len(self.spectrum.channels)-1):
            # If above threshold find the domain
            if self.conv_n_sigma_spectrum[ind] > self.n_sigma_threshold:
                peak_domain = self.peaks_domain(self.spectrum.energy_calibration(channel))
                if len(domains) == 0:
                    domains.append(peak_domain)
                elif peak_domain[0] <= domains[-1][0]:
                    domains[-1] = (domains[-1][0], peak_domain[1])
                else:
                    domains.append(peak_domain)
                channel = peak_domain[1]
                ind = np.abs(self.spectrum.channels - peak_domain[1]).argmin()
            channel = channel + 1
            ind = ind + 1
        return domains

    def to_peak(self, value_in_domain):
        """
        find and return the peak in which the value_in_domain is in.

        Parameters
        ----------
        value_in_domain: float
        value in the domain which is inside a peak.

        Returns
        -------
        Peak
        the peak in which value in domain is in

        """
        spectrum = self.spectrum.counts
        peak_domain = self.peaks_domain(value_in_domain)
        # fwhm and center of the peak
        middle_channel = round((peak_domain[0] + peak_domain[1]) / 2)
        ch_fwhm = self.spectrum.fwhm_calibration(
            self.spectrum.energy_calibration(middle_channel)) / self.spectrum.energy_calibration[1]

        # the background levels from left and right
        peak_bg_l = ufloat(spectrum[peak_domain[0] - round(ch_fwhm / 2):peak_domain[0]].mean(),
                           spectrum[peak_domain[0] - round(ch_fwhm / 2):peak_domain[0]].std())
        peak_bg_r = ufloat(spectrum[peak_domain[1]:peak_domain[1] + round(ch_fwhm / 2)].mean(),
                           spectrum[peak_domain[1]:peak_domain[1] + round(ch_fwhm / 2)].std())
        # the peak
        peak = self.spectrum.xr_spectrum().sel(
            energy=slice(self.spectrum.energy_calibration(peak_domain[0]),
                         self.spectrum.energy_calibration(peak_domain[1])))
        # fit the peak using the fitting method
        return Peak(peak, peak_bg_l, peak_bg_r)


