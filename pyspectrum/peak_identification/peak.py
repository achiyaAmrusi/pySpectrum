# NEED TO DELETE SOME METHODS, THE FIT METHOD CAN HAVE ONE FUNCTION WITH THREE OUTPUTS
import xarray as xr
import numpy as np
import math
from uncertainties import ufloat, nominal_value, std_dev
from pyspectrum.peak_identification.peaks_fit import GaussianWithBGFitting


class Peak:
    """
        Represents a peak with methods for 1D peak gaussian fit, center calculation and so on...

    Parameters
    ----------
    peak_xarray: xr.DataArray
     The peak counts and energies in form of an xarray
    ubackground_l, ubackground_r: ufloat (default ufloat(0, 1))
     Mean counts from the left and right to the peak
     This is needed for background subtraction
    Attributes
    ----------
    peak_xarray: xr.DataArray
     The peak counts and energies in form of an xarray
    height_left, height_right: ufloat (default ufloat(0, 1))
     Mean counts from the left and right to the peak
     This is needed for background subtraction
    estimated_center, estimated_resolution: float
     the estimated mean and resolution
    Methods
    -------

    - `__init__(self, xarray.DataArray)`:
      Constructor method to initialize a Peak instance.

    - `peak_gaussian_fit_parameters(self, energy_in_the_peak, resolution_estimation=1, background_subtraction=False)`:
      Fit a Gaussian function to a peak in the spectrum and return fit parameters.

    - `peak_energy_center_first_moment_method(self, energy_in_the_peak, detector_energy_resolution=1,
                                              background_subtraction=False)`:
      Calculate the center (mean) of a peak using the first moment method.

    - `counts_in_fwhm_sum_method(self, energy_in_the_peak, detector_energy_resolution=1)`:
      Calculate the sum of counts within the Full Width at Half Maximum (FWHM) of a peak.

    - `counts_in_fwhm_fit_method(self, energy_in_the_peak, detector_energy_resolution=1)`:
      Calculate the sum of counts within the FWHM using a fit-based method.

    - 'subtract_background(self):
        subtract background from the peak
        """

    # Constructor method
    def __init__(self, peak_xarray: xr.DataArray, ubackground_l=ufloat(0, 1), ubackground_r=ufloat(0, 1)):
        """ Constructor of peak.


        """
        # Instance variables
        if not (isinstance(peak_xarray, xr.DataArray)) and len(peak_xarray.dims) == 1:
            raise TypeError("Variable peak_xarray must be of type 1d xr.DataArray.")
        # peak variable is channel
        self.peak = peak_xarray.rename({peak_xarray.dims[0]: 'channel'})
        # initialization
        _, self.estimated_center, self.estimated_resolution = GaussianWithBGFitting.gaussian_initial_guess_estimator(self.peak)

        if not (isinstance(ubackground_l, type(ufloat(0, 0)))):
            raise TypeError("Variable ubackground_l must be of type ufloat.")
        self.height_left = ubackground_l

        if not (isinstance(ubackground_r, type(ufloat(0, 0)))):
            raise TypeError("Variable ubackground_r must be of type ufloat.")
        self.height_right = ubackground_r


    def center_fwhm_estimator(self):
        """ calculate the center of the peak
        the function operate by the following order -
         calculate the peak domain,
          find maximal value
          define domain within fwhm edges
          calculate the mean energy which is the center of the peak (like center of mass)
        Returns
        ד-------
          """
        peak = self.peak
        maximal_count = peak.max()
        # Calculate the half-maximum count
        half_max_count = maximal_count / 2

        # Find the energy values at which the counts are closest to half-maximum on each side
        minimal_channel = peak.where(peak >= half_max_count, drop=True)['channel'].to_numpy()[0]
        maximal_channel = peak.where(peak >= half_max_count, drop=True)['channel'].to_numpy()[-1]
        # define the full width half maximum area (meaning the area which is bounded by the fwhm edges)
        fwhm_slice = peak.sel(channel=slice(minimal_channel, maximal_channel))
        # return the mean energy in the fwhm which is the energy center
        return (fwhm_slice * fwhm_slice.coords['channel']).sum() / fwhm_slice.sum(), (maximal_channel - minimal_channel)

    def peak_with_errors(self):
        """Return Peak. peak in ufloat for each count, the errors are in (counts)**0.5 format
        Returns
        -------
        xr.DataArray: Xarray representation of the spectrum with ufloat as values.
        """
        counts_with_error = [ufloat(count, abs(count) ** 0.5) for count in self.peak.values]
        return xr.DataArray(counts_with_error, coords=self.peak.coords, dims=['channel'])

    def sum_method_counts_under_fwhm(self):
        """Calculate the sum of counts within the Full Width at Half Maximum (FWHM) of a peak.

        The function operates by the following steps:
        1. Calculate the FWHM of the specified peak using the `peak_fwhm_fit_method`.
        2. Estimate the center of the peak using the `peak_energy_center_first_moment_method`.
        3. Define the energy domain within the FWHM edges.
        4. Slice the spectrum to obtain the counts within the FWHM domain.
        5. Return the sum of counts within the FWHM and its associated uncertainty.

        Returns
        -------
        ufloat: A ufloat containing the sum of counts within the FWHM and its associated uncertainty.
          The uncertainty is calculated using uncertainties package

        Note: The function assumes background subtraction is performed during FWHM calculation.
        """
        # Calculate the peak domain and slice the peak

        fit_parameter = self.gaussian_fit_parameters()
        fwhm = fit_parameter['curvefit_coefficients'].sel(param='fwhm')
        peak_mean = fit_parameter['curvefit_coefficients'].sel(param='mean')
        minimal_channel = peak_mean - fwhm / 2
        maximal_channel = peak_mean + fwhm / 2
        energy = self.peak.coords['channel']
        center_index = np.where(energy > peak_mean)[0][0]
        de = energy[center_index + 1] - energy[center_index]
        fwhm_slice = (
            self.subtract_background(with_errors=True)).sel(channel=slice(minimal_channel - de / 2, maximal_channel))
        # return counts under fwhm
        return fwhm_slice.sum()

    def gaussian_fit_parameters(self, background_subtraction=True):
        """Fit a Gaussian function to a peak in the spectrum and return fit parameters.

        The function fits a Gaussian function to the specified peak in the spectrum. The peak's location and
        resolution are estimated, and if background_subtraction is enabled, background subtraction is performed
        before the fitting process.

        Parameters:
        - energy_in_the_peak (float): The energy in the peak which to fit the Gaussian peak.
        - resolution_estimation (float, optional): The estimated resolution of the peak (not of the detector,
        for example in doppler broadening.) Default is 1.
        - background_subtraction (bool, optional): If True, subtract background before fitting. Default is False.

        Returns
        -------
        - tuple: A tuple containing the fit parameters and covariance matrix of the Gaussian fit.
        The fit parameters include:
        - Amplitude: Amplitude of the Gaussian peak.
        - Center: Center (mean) of the Gaussian peak.
        - FWHM: Full Width at Half Maximum of the Gaussian peak.

        The covariance matrix provides the uncertainties in the fit parameters.
        """
        fit_params = GaussianWithBGFitting.single_gaussian_fitting(self.peak, [self.height_left - self.height_right,
                                                                               self.height_right])
        return fit_params[0]

    def fit_method_counts_under_fwhm(self):
        """Calculate the sum of counts within the Full Width at Half Maximum (FWHM) of a peak.

        The function operates by the following steps:
        1. Estimate the amplitude of the specified peak using the `peak_amplitude_fit_method`.
        2. Estimate the FWHM of the specified peak using the `peak_fwhm_fit_method`.
        3. using the formula for the counts to return the counts number
        the formula is  0.761438079*A*np.sqrt(2 *pi)*(fwhm/(2*np.sqrt(2*np.log(2))))* 1/bin_energy_size
        where
        - 0.761438079 the area under the fwhm of a standard gaussian A*exp(-x**2/sigma)
        - A gaussian amplitude
        - np.sqrt(2 *pi)*(fwhm/(2*np.sqrt(2*np.log(2)))) amplitude correction (change of variable)
        - 1/bin_energy_size change of variable (The amplitude depends on the bin width)

        Parameters:
        - energy_in_the_peak (float): The energy around which to calculate the FWHM and sum counts.
        - detector_energy_resolution (float, optional): The resolution of the detector. Default is 1.

        Returns
        -------
        tuple: A tuple containing the sum of counts within the FWHM and its associated uncertainty.
          The uncertainty is calculated considering the covariance matrix obtained from the Gaussian fit,
          accounting for the covariance of the other fit parameters up to one standard deviation in them.

        Note: The function assumes background subtraction is performed during FWHM calculation.
        """
        # Calculate the peak amplitude and fwhm
        fit_parameter = self.gaussian_fit_parameters()
        amplitude = fit_parameter['curvefit_coefficients'].sel(param='amplitude')
        amplitude_error = fit_parameter['curvefit_covariance'].sel(cov_i='amplitude', cov_j='amplitude')
        fwhm = fit_parameter['curvefit_coefficients'].sel(param='fwhm')
        # energy size of each bin
        bin_size = self.peak.coords['channel'][1] - self.peak.coords['channel'][0]
        # factor to get area under the fwhm
        factor_of_area = 0.76096811 * np.sqrt(2 * np.pi) * (fwhm / (2 * np.sqrt(2 * np.log(2))))
        return ufloat(factor_of_area * amplitude * (1 / bin_size), factor_of_area * amplitude_error * (1 / bin_size))

    def first_moment_method_center(self, background_subtraction=False):
        """Calculate the center (mean) of a peak in the spectrum.

        The function estimates the center of the specified peak by finding the full width half maximum domain and
        that it use the mean on the spectrum slice to calculate the peak center .
        If background_subtraction is enabled, background subtraction is performed before the fitting process.

        Parameters:
        - energy_in_the_peak (float): The energy around which to calculate the peak center.
        - detector_energy_resolution (float, optional): The energy resolution of the peak. Default is 1.
        - background_subtraction (bool, optional): If True, subtract background before fitting. Default is False.

        Returns
        -------
        ufloat: A ufloat containing the peak center and its associated uncertainty.
          The uncertainty is calculated using uncertainties package

        Note: The function assumes the output of `peak_gaussian_fit_parameters` is used for fitting.
        """
        # Calculate the peak domain and slice the peak
        fit_parameter = self.gaussian_fit_parameters()
        fwhm = fit_parameter['curvefit_coefficients'].sel(param='fwhm')
        fwhm_error = fit_parameter['curvefit_covariance'].sel(cov_i='fwhm', cov_j='fwhm')
        peak_mean = fit_parameter['curvefit_coefficients'].sel(param='mean')
        peak_mean_error = fit_parameter['curvefit_covariance'].sel(cov_i='mean', cov_j='mean')
        if not (math.isnan(fwhm_error) or math.isnan(peak_mean_error)):
            minimal_channel = max(peak_mean - peak_mean_error - fwhm - fwhm_error,
                                  self.peak.coords['channel'][0])
            maximal_channel = min(peak_mean + peak_mean_error + fwhm + fwhm_error,
                                  self.peak.coords['channel'][-1])
            fwhm_slice = (self.peak_with_errors()).sel(channel=slice(minimal_channel, maximal_channel))
        else:
            minimal_channel = max(peak_mean - fwhm,
                                  self.peak.coords['channel'][0])
            maximal_channel = min(peak_mean + fwhm,
                                  self.peak.coords['channel'][-1])
            fwhm_slice = (self.peak_with_errors()).sel(channel=slice(minimal_channel, maximal_channel))
        # return the mean energy in the fwhm which is the energy center
        center = (fwhm_slice * fwhm_slice.coords['channel']).sum() / fwhm_slice.sum()
        return nominal_value(center.values.item()), std_dev(center.values.item())

    def subtract_background(self, with_errors=False):
        """ create a subtracted background pyspectrum from the xarray pyspectrum
        the method of the subtraction is as follows -
         - find the peak domain and the mean count in the edges
        - calculated erf according to the edges
        - subtract edges from pyspectrum  """
        if not with_errors:
            peak = self.peak
            height_left = nominal_value(self.height_left)
            height_right = nominal_value(self.height_right)
        else:
            peak = self.peak_with_errors()
            height_left = self.height_left
            height_right = self.height_right
        # define new coordinates in resolution units (for the error function to be defined correctly)
        peak = peak.assign_coords(channel=(peak.coords['channel'] - self.estimated_center) / self.estimated_resolution)
        # define error function on the domain
        erf_background = np.array([(math.erf(-x) + 1) for x in peak.coords['channel'].to_numpy()])
        # subtraction
        peak_no_bg = (peak - 0.5 * (height_left - height_right) * erf_background - height_right)
        # return the peak with the original coordinates
        return peak_no_bg.assign_coords(channel=self.peak.coords['channel'])
