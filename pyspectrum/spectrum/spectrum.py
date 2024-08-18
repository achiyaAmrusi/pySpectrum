import xarray as xr
import numpy as np
import pandas as pd
from uncertainties import ufloat
from pyspectrum.peak_identification.find_gamma_peaks import FindPeaks
from pyspectrum.peak_identification.convolution import Convolution
from pyspectrum.peak_identification.new_peaks_fit import  GaussianWithBGFitting
from pyspectrum.peak_identification.zero_area_functions import gaussian_2_dev


class Spectrum:
    """
        Represents a spectrum with methods for data manipulation and analysis.

        Parameters
        ----------
        counts : np.ndarray
         1D array of spectrum counts.
        channels : np.ndarray
         1D array of spectrum channels.
        energy_calibration_poly : np.poly1d
         Calibration polynomial for energy calibration.
        fwhm_calibration : Callable
         a given method that return the fwhm per energy

        Attributes
        ----------
        counts (numpy.ndarray): Array of counts.
        channels (numpy.ndarray): Array of channels.
        energy_calibration (numpy.poly1d): Polynomial for energy calibration.
        fwhm_calibration (function): function for fwhm calibration channel->fwhm.

        Methods
        -------
        `__init__(self, counts, channels, energy_calibration_poly=np.poly1d([1, 0]))`
        Constructor method to initialize a Spectrum instance.

        `xr_spectrum(self, errors=False)`
        Returns the spectrum in xarray format. If errors is True, the xarray values will be in ufloat format.

        `calibrate_energy(self, energy_calibration)`
        change the energy calibration polynom of Spectrum

        `calibrate_fwhm(self, fwhm_calibration)`
        change the fwhm calibration polynom of Spectrum

        `from_file(file_path, energy_calibration_poly=np.poly1d([1, 0]), fwhm_calibration=None, sep='\t',
                  **kwargs)`
        load spectrum from a file which has 2 columns,
        first column is the channels/energy and the second is counts, in te end the function return Spectrum
        """

    # Constructor method
    def __init__(self, counts, channels, energy_calibration_poly=np.poly1d([1, 0]), fwhm_calibration=None):
        # Instance variables
        if not (isinstance(counts, np.ndarray) and counts.ndim == 1):
            raise TypeError("Variable counts must be of type 1 dimension np.array.")
        self.counts = counts
        if not (isinstance(channels, np.ndarray) and channels.ndim == 1):
            raise TypeError("Variable channels must be of type 1 dimension np.array.")
        self.channels = channels
        if not isinstance(energy_calibration_poly, np.poly1d):
            raise TypeError("Variable energy_calibration_poly must be of type numpy.poly1d.")
        self.energy_calibration = energy_calibration_poly
        self.fwhm_calibration = fwhm_calibration
        self._convolution = None
        self._find_peaks = None

    def xr_spectrum(self):
        """
        Return pyspectrum in xarray format
        If errors is True, the xarray values will be in ufloat format.

        Parameters
        ----------
        errors (bool): If True, the xarray values will be in ufloat format, including counts error(no option to
        time normalize yet).

        Returns
        -------
        xr.DataArray: Xarray representation of the spectrum.
        """
        spectrum = xr.DataArray(self.counts, coords={'energy': self.energy_calibration(self.channels)},
                                dims=['energy'])
        return spectrum

    def calibrate_energy(self, energy_calibration):
        """
        change the energy calibration polynom of Spectrum

        Parameters
        ----------
        energy_calibration: np.poly1d
         The calibration function energy_calibration(channel) -> detector energy.
        """
        if not isinstance(energy_calibration, np.poly1d):
            raise TypeError("Variable x must be of type numpy.poly1d.")
        self.energy_calibration = energy_calibration

    def calibrate_fwhm(self, fwhm_calibration):
        """
        change the energy calibration polynom of Spectrum

        Parameters
        ----------
         fwhm_calibration: Callable
          The calibration function fwhm_calibration(channel) -> fwhm in the channel.
         """
        if not callable(fwhm_calibration):
            raise TypeError("fwhm_calibration needs to be callable.")
        self.fwhm_calibration = fwhm_calibration

    @staticmethod
    def from_file(file_path, energy_calibration_poly=np.poly1d([1, 0]), fwhm_calibration=None, sep='\t',
                  **kwargs):
        """
        load spectrum from a file which has 2 columns which tab between them
        first column is the channels/energy and the second is counts
        function return Spectrum
        Parameters
        ----------
        file_path: str
         two columns with tab(\t) between them. first line is column names - channel, counts
        energy_calibration_poly: numpy.poly1d([a, b])
         the energy calibration of the detector
        fwhm_calibration: Callable
         a function that given energy/channel(first raw in file) returns the fwhm
        sep: str
         the separation letter
        kwargs: more parameter for pd.read_csv

        Returns
        -------
        Spectrum
        the spectrum from the files with the given parameters
        """
        # Load the pyspectrum file in form of DataFrame
        try:
            data = pd.read_csv(file_path, sep=sep, names=['channel', 'counts'], **kwargs)
        except ValueError:
            raise FileNotFoundError(f"The given data file path '{file_path}' do not exist.")
        return Spectrum.from_dataframe(data, energy_calibration_poly, fwhm_calibration)

    @staticmethod
    def from_dataframe(spectrum_df, energy_calibration_poly=np.poly1d([1, 0]), fwhm_calibration=None):
        """
        load spectrum from a file which has 2 columns which tab between them
        first column is the channels/energy and the second is counts
        function return Spectrum
        Parameters
        ----------
        spectrum_df: pd.DataFrame
         spectrum in form of a dataframe such that the column are -  'channel', 'counts'
        energy_calibration_poly: numpy.poly1d([a, b])
        the energy calibration of the detector
         fwhm_calibration:a function that given energy/channel(first raw in file) returns the fwhm
        Returns
        -------
        Spectrum
        the spectrum from the files with the given parameters
        """
        # Load the pyspectrum file in form of DataFrame

        return Spectrum(spectrum_df['counts'].to_numpy(),
                        spectrum_df['channel'].to_numpy(),
                        energy_calibration_poly,
                        fwhm_calibration)

    def fit_peaks(self, fitting_method=GaussianWithBGFitting(), zero_area_function=gaussian_2_dev, n_sigma_threshold=4,
                  refind_peaks_flag=False):
        """
        The function finds and fit peaks automatically.
        function scan the spectral_n_sigma vector which is kernel_convolution/kernel_convolution_std
        and search for when the value is larger than n_sigma_threshold.
         if the value is above n_sigma_threshold, it calls peak_domain.
        the peaks are than fitted and returned as Peaks and Peaks properties

        Parameters
        ----------
        fitting_method: fitPeakFit
        class of fitting method as in peaks which can fit the peaks
        zero_area_function: callable
        function from pyspectrum.peak_identification.zero_area_functions
        n_sigma_threshold: float
        the signal-to-noise ratio cutoff for peak identification. For rough recognition in gamma spectroscopy 4 is good choice.
        refind_peaks_flag: bool
        to recalculate findpeaks
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
        # initialize the find peaks if it wasn't initialized yet
        # Not that right now it is problematic to refind peaks with different functions
        if (self._convolution is None or self._find_peaks is None) or refind_peaks_flag:
            self._convolution = Convolution(self.fwhm_calibration, zero_area_function)
            self._find_peaks = FindPeaks(self, self._convolution, n_sigma_threshold)
        peaks_domain = self._find_peaks.find_domains_above_snr()
        spectrum = self.counts
        peaks_properties = []
        peaks_valid_domain = []
        for peak_domain in peaks_domain:
            # fwhm in the peak
            middle_channel = round((peak_domain[0] + peak_domain[1]) / 2)
            ch_fwhm = self.fwhm_calibration(
                self.energy_calibration(middle_channel)) / self.energy_calibration[1]

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
            peak = self.xr_spectrum().sel(
                energy=slice(self.energy_calibration(peak_domain[0]),
                             self.energy_calibration(peak_domain[1])))
            # fit the peak using the fitting method
            peak_background_data = [peak_bg_l - peak_bg_r, peak_bg_r]

            fit_properties = fitting_method.peak_fit(peak, peak_background_data)

            # save all the peak found in the domain
            if fit_properties:
                peaks_properties.append(fit_properties)
                peaks_valid_domain.append(peak_domain)
        return peaks_properties, peaks_valid_domain

    def plot_all_peaks(self, fitting_method=GaussianWithBGFitting(), zero_area_function=gaussian_2_dev, n_sigma_threshold=4, refind_peaks_flag=False):
        """plot the peaks found in find_peaks via fitting method

        Parameters
        ----------
        fitting_method: fitPeakFit
        class of fitting method as in peaks which can fit the peaks
        zero_area_function: callable
        function from pyspectrum.peak_identification.zero_area_functions
        n_sigma_threshold: float
        the signal-to-noise ratio cutoff for peak identification. For rough recognition in gamma spectroscopy 4 is good choice.
        refind_peaks_flag: bool
        to recalculate findpeaks
       """
        peaks_properties, peaks_domain = self.fit_peaks(fitting_method, zero_area_function, n_sigma_threshold, refind_peaks_flag)
        self.xr_spectrum().plot()
        for i, peak in enumerate(peaks_properties):
            energy_domain = self.energy_calibration(np.arange(peaks_domain[i][0], peaks_domain[i][1], 1))
            fitting_method.plot_fit(energy_domain, peak)
