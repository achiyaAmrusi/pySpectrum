"""
Module for handling spectral data.

This module defines the Spectrum class for representing and processing spectral data.
"""


import xarray as xr
import numpy as np
import pandas as pd
from uncertainties import ufloat


class Spectrum:
    """
        Represents a spectrum with methods for data manipulation and analysis.

        Attributes:
        __________
        counts (numpy.ndarray): Array of counts.
        channels (numpy.ndarray): Array of channels.
        energy_calibration (numpy.poly1d): Polynomial for energy calibration.
        fwhm_calibration (function): function for fwhm calibration channel->fwhm.

        Methods:
        ________

        `__init__(self, counts, channels, energy_calibration_poly=np.poly1d([1, 0]))`
        Constructor method to initialize a Spectrum instance.

        `xr_spectrum(self, errors=False)`
        Returns the spectrum in xarray format. If errors is True, the xarray values will be in ufloat format.

        `calibrate_energy(self, energy_calibration)`
        change the energy calibration polynom of Spectrum

        `calibrate_fwhm(self, fwhm_calibration)`
        change the fwhm calibration polynom of Spectrum

        `load_spectrum_file_to_spectrum_class(cls, file_path,
                                             energy_calibration_poly=np.poly1d([1, 0]),
                                              fwhm_calibration=None, sep='\t', **kwargs)`
        load spectrum from a file which has 2 columns,
        first column is the channels/energy and the second is counts, in te end the function return Spectrum

        """

    # Constructor method
    def __init__(self, counts, channels, energy_calibration_poly=np.poly1d([1, 0]), fwhm_calibration=None):
        """ Constructor of Spectrum.

        Parameters
        ----------
        counts (np.ndarray): 1D array of spectrum counts.
        channels (np.ndarray): 1D array of spectrum channels.
        energy_calibration_poly (np.poly1d): Calibration polynomial for energy calibration.
        fwhm_calibration (function): a given method
        """
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

    # Instance method
    def xr_spectrum(self, errors=False):
        """Return pyspectrum in xarray format
        If errors is True, the xarray values will be in ufloat format.
        Parameters
        ----------
        errors (bool): If True, the xarray values will be in ufloat format, including counts error(no option to
        time normalize yet).

        Returns
        -------
        xr.DataArray: Xarray representation of the spectrum.
        """
        if not errors:
            spectrum = xr.DataArray(self.counts, coords={'energy': self.energy_calibration(self.channels)},
                                    dims=['energy'])
        else:
            counts_with_error = [ufloat(count, abs(count) ** 0.5) for count in self.counts]
            spectrum = xr.DataArray(counts_with_error, coords={'energy': self.energy_calibration(self.channels)},
                                    dims=['energy'])
        return spectrum

    def calibrate_energy(self, energy_calibration):
        """
        change the energy calibration polynom of Spectrum
        Parameters
        ----------
        energy_calibration (np.poly1d): The calibration function energy_calibration(channel) -> detector energy.

        Returns
        -------
        Nothing.
                """
        if not isinstance(energy_calibration, np.poly1d):
            raise TypeError("Variable x must be of type numpy.poly1d.")
        self.energy_calibration = energy_calibration

    def calibrate_fwhm(self, fwhm_calibration):
        """change the energy calibration polynom of Spectrum
        Parameters
        ----------
         - fwhm_calibration (function): The calibration function fwhm_calibration(channel) -> fwhm in the channel.

        Returns
        -------
        Nothing.
                """
        if not callable(fwhm_calibration):
            raise TypeError("fwhm_calibration needs to be callable.")
        self.fwhm_calibration = fwhm_calibration


    @classmethod
    def load_spectrum_file_to_spectrum_class(cls, file_path,
                                             energy_calibration_poly=np.poly1d([1, 0]),
                                             fwhm_calibration=None, sep='\t', **kwargs):

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
        fwhm_calibration:a function that given energy/channel(first raw in file) returns the fwhm
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
            data = pd.read_csv(file_path, sep=sep, **kwargs)
        except ValueError:
            raise FileNotFoundError(f"The given data file path '{file_path}' do not exist.")
        return Spectrum(data[data.columns[1]].to_numpy(),
                        data[data.columns[0]].to_numpy(),
                        energy_calibration_poly,
                        fwhm_calibration)
