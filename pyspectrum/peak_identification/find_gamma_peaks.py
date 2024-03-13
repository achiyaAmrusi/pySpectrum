import numpy as np
from pyspectrum.spectrum import Spectrum
from pyspectrum.peak_identification.peaks_fit import GaussianWithBGFitting
from pyspectrum.peak_identification.convolution import Convolution
from pyspectrum.peak_identification.peak import Peak
from uncertainties import ufloat

EPSILON = 1e-4



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
        fitting_type: str
        the type of the fitting method. for now there is only
        'HPGe_spectroscopy'
        Returns
        -------
"""
        self.spectrum = spectrum
        self.convolution = convolution
        _, _, self.conv_n_sigma_spectrum = convolution.convolution(spectrum.energy_calibration(spectrum.channels),
                                                                   spectrum.counts)
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
        peak properties  =  properties of fit curve in xarray

        References
        ----------
        [1]Phillips, Gary W., and Keith W. Marlow.
         "Automatic analysis of gamma-ray spectra from germanium detectors."
          Nuclear Instruments and Methods 137.3 (1976): 525-536.
        [2]Likar, Andrej, and Tim Vidmar.
        "A peak-search method based on spectrum convolution."
        Journal of Physics D: Applied Physics 36.15 (2003): 1903.
        """

        peaks_domain = self.find_domains_above_snr()
        spectrum = self.spectrum.counts
        peaks_properties = []
        peaks_valid_domain = []
        fit_worked = True
        for peak_domain in peaks_domain:
            # fwhm in the peak
            middle_channel = round((peak_domain[0] + peak_domain[1]) / 2)
            ch_fwhm = self.spectrum.fwhm_calibration(
                self.spectrum.energy_calibration(middle_channel)) / self.spectrum.energy_calibration[1]

            # the background levels from left and right
            if (peak_domain[0] - round(ch_fwhm / 2)) < 0:
                peak_domain_left = peak_domain[0] if peak_domain[0] > 0 else peak_domain[0]+1
                peak_bg_l = ufloat(spectrum[0:peak_domain_left].mean(),
                                   spectrum[0:peak_domain_left].std())
            else:
                peak_bg_l = ufloat(spectrum[peak_domain[0] - round(ch_fwhm / 2):peak_domain[0]].mean(),
                                   spectrum[peak_domain[0] - round(ch_fwhm / 2):peak_domain[0]].std())

            if (peak_domain[1] + round(ch_fwhm / 2)) > len(spectrum):
                peak_domain_right = peak_domain[1] if peak_domain[1] < len(spectrum) else peak_domain[1]-1
                peak_bg_r = ufloat(spectrum[peak_domain_right:len(spectrum)].mean(),
                                   spectrum[peak_domain_right:len(spectrum)].std())
            else:
                peak_bg_r = ufloat(spectrum[peak_domain[1]:peak_domain[1] + round(ch_fwhm / 2)].mean(),
                                   spectrum[peak_domain[1]:peak_domain[1] + round(ch_fwhm / 2)].std())

            # the peak
            peak = self.spectrum.xr_spectrum().sel(
                energy=slice(self.spectrum.energy_calibration(peak_domain[0]),
                             self.spectrum.energy_calibration(peak_domain[1])))
            # fit the peak using the fitting method
            peak_background_data = [peak_bg_l - peak_bg_r, peak_bg_r]
            fit_properties, num_of_peaks = self.fitting_method.peak_fit(peak, peak_background_data)
            # save all the peak found in the domain
            if fit_properties:
                for fit in fit_properties:
                    peaks_properties.append(fit)
                peaks_valid_domain.append(peak_domain)

        return peaks_properties, peaks_valid_domain

    def find_domains_above_snr(self):
        """
        The function finds peaks domain automatically
        function scan the spectral_n_sigma vector which is kernel_convolution/kernel_convolution_std
        and search for when the value is larger than n_sigma_threshold, if the value is above n_sigma_threshold,
        it checks it finds the peak domain using self.peaks_domain
        Warning, function does not distinguish if the peak is a valid peak, only if it is above noise.
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
        """

        domains = []
        ind = 1
        channel = self.spectrum.channels[1]
        while ind < (len(self.spectrum.channels)-1):
            # If above threshold find the domain
            if self.conv_n_sigma_spectrum[ind] > self.convolution.n_sigma_threshold:
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

    def plot_all_peaks(self):
        """plot the peaks found in find_peaks via fitting method
        Parameters
        ----------

        Returns
        -------

   """
        peaks_properties, peaks_domain = self.find_all_peaks()
        self.spectrum.xr_spectrum().plot()
        for i, peak in enumerate(peaks_properties):
            energy_domain = self.spectrum.energy_calibration(np.arange(peaks_domain[i][0], peaks_domain[i][1], 1))
            self.fitting_method.plot_fit(energy_domain, peak)
