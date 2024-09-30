# NEED TO DELETE SOME METHODS, THE FIT METHOD CAN HAVE ONE FUNCTION WITH THREE OUTPUTS
import xarray as xr
import numpy as np
import math
from uncertainties import ufloat, nominal_value
from uncertainties.unumpy import nominal_values, std_devs, uarray
from pyspectrum.peak_fitting.std_gaussian_fitting import GaussianWithBGFitting


class Peak:
    """
    Represents a peak with methods for 1D peak gaussian fit, center calculation and so on.
    All the uncertainties are calculated using uncertainties package

    Parameters
    ----------
    peak_xarray: xr.DataArray
     The peak counts and energies in form of  xarray.DataArray
    ubackground_l: ufloat (default ufloat(0, 1))
    ubackground_r: ufloat (default ufloat(0, 1))
     Mean counts from the left and right to the peak
     This is needed for background subtraction
    Attributes
    ----------
    peak: xr.DataArray
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

    - `center_fwhm_estimator(peak: xr.DataArray)`:
        an initial estimation for single peak mean and fwhm

    - `direct_sum_counts_under_fwhm(self, number_of_fwhm=1):
      Calculate the sum of counts within the Full Width at Half Maximum (FWHM) of a peak.
      There is option to sm over few FWHM

    - `gaussian_fit_parameters(self)`:
      Fit a Gaussian function to a peak in the spectrum and return fit parameters.

    - `fit_method_counts_under_fwhm(self)`:
      Calculate the sum of counts within the Full Width at Half Maximum (FWHM) of a peak uding fitting method/

    - `first_moment_method_center(self)`:
        Calculate the center (mean) of a peak in the spectrum

    - 'subtract_background(self):
        create a subtracted background xarray of the peak
        """

    # Constructor method
    def __init__(self, peak_xarray: xr.DataArray, ubackground_l=ufloat(0, 1), ubackground_r=ufloat(0, 1)):
        # Instance variables
        if not (isinstance(peak_xarray, xr.DataArray)) and len(peak_xarray.dims) == 1:
            raise TypeError("Variable peak_xarray must be of type 1d xr.DataArray.")
        # peak variable is channel
        self.peak = peak_xarray.rename({peak_xarray.dims[0]: 'channel'})
        # initialization
        self.estimated_center, self.estimated_resolution = Peak.center_fwhm_estimator(self.peak)

        if not (isinstance(ubackground_l, type(ufloat(0, 0)))):
            raise TypeError("Variable ubackground_l must be of type ufloat.")
        self.height_left = ubackground_l

        if not (isinstance(ubackground_r, type(ufloat(0, 0)))):
            raise TypeError("Variable ubackground_r must be of type ufloat.")
        self.height_right = ubackground_r

    @staticmethod
    def center_fwhm_estimator(peak: xr.DataArray):
        """
        estimate the center of the peak
        the function operate by the following order -
        1. find maximal value
        2. define domain within fwhm edges
        3. calculate the mean energy which is the center of the peak (like center of mass)

        Returns
        -------
        tuple
            center, fwhm

          """
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

    def direct_sum_counts_under_fwhm(self, number_of_fwhm=1):
        """
        Calculate the sum of counts within the Full Width at Half Maximum (FWHM) of a peak.

        The function operates by the following steps:
        1. Takes the fwhm to be self.estimated_resolution
        2. Estimate the center of the peak using the `peak_energy_center_first_moment_method`.
        3. Define the energy domain within the FWHM edges.
        4. calculate the background subtracted peak
        4. Slice the spectrum to obtain the counts within the FWHM domain and sum over it.

        Returns
        -------
        ufloat: A ufloat containing the sum of counts within the FWHM and its associated uncertainty.
          The uncertainty is calculated using uncertainties package

        Note: The function assumes background subtraction is performed during FWHM calculation.
        """

        fwhm = self.estimated_resolution
        peak_mean = nominal_value(self.first_moment_method_center())
        minimal_channel = peak_mean - number_of_fwhm * (fwhm / 2)
        maximal_channel = peak_mean + number_of_fwhm * (fwhm / 2)
        energy = self.peak.coords['channel']
        center_index = np.where(energy > peak_mean)[0][0]
        de = energy[center_index + 1] - energy[center_index]
        fwhm_slice = (self.subtract_background()).sel(channel=slice(minimal_channel - de / 2, maximal_channel))
        # return counts under fwhm
        return fwhm_slice.sum()

    def gaussian_fit_parameters(self):
        """
        Fit a Gaussian function to a peak in the spectrum and return fit parameters.

        Returns
        -------
        xr.DataSet
            Gaussian plus background parameters values and uncertainties.
        """
        fit_params = GaussianWithBGFitting.fit(self.peak, peaks_centers=[self.estimated_center],
                                               estimated_fwhm=self.estimated_resolution,
                                               background_parameters=[self.height_left - self.height_right,
                                                                      self.height_right])
        return fit_params

    def fit_method_counts_under_fwhm(self):
        """
        Calculate the sum of counts within the Full Width at Half Maximum (FWHM) of a peak uding fitting method.

        The function operates by the following steps:
        1. Estimate the amplitude and fwhm of the specified peak using fitting
        2. using the formula for the counts to return the counts number
        the formula is  0.761438079*A*np.sqrt(2 *pi)*(fwhm/(2*np.sqrt(2*np.log(2))))* 1/bin_energy_size
        where
        - 0.761438079 the area under the fwhm of a standard gaussian A*exp(-x**2/sigma)
        - A gaussian amplitude
        - np.sqrt(2 *pi)*(fwhm/(2*np.sqrt(2*np.log(2)))) amplitude correction (change of variable)
        - 1/bin_energy_size change of variable (The amplitude depends on the bin width)

        Returns
        -------
        tuple
         peak fwhm, uncertainty
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

    def first_moment_method_center(self):
        """
        Calculate the center (mean) of a peak in the spectrum using the mean over the fwhm

        Returns
        -------
        tuple
         peak center, uncertainty
        """
        # extract the slice of 2 fwhm width from each side of the spectrum
        minimal_channel = max(self.estimated_center - 2*self.estimated_resolution, self.peak.coords['channel'][0])
        maximal_channel = min(self.estimated_center + 2*self.estimated_resolution, self.peak.coords['channel'][-1])
        fwhm_slice = (self.subtract_background()).sel(channel=slice(minimal_channel, maximal_channel))

        # calculate the mean energy in the fwhm which is the energy center
        center = (fwhm_slice * fwhm_slice.coords['channel']).sum() / fwhm_slice.sum()
        return center.values.item()

    def subtract_background(self):
        """
        create a subtracted background xarray of the peak
        the background is assumed to be an erf with std of the peak and the center of the peak.

        Returns
        -------
        xr.DataArray
        the peak with subtracted background and errors of poisson
        """
        # background estimation
        std = (self.estimated_resolution / (2 * np.sqrt(2 * np.log(2))))
        erf = np.array([(math.erf(-(x - self.estimated_center) / std) + 1) for x in self.peak.coords['channel'].to_numpy()])
        height_difference = self.height_left-self.height_right
        peak_baseline = min(self.height_right, self.height_left)
        approx_bg = 0.5 * height_difference * erf + peak_baseline

        # background subtraction
        approx_peak = self.peak - nominal_values(approx_bg)
        # poisson error
        approx_std_peak = abs(approx_peak)**0.5
        # into an xarray
        peak_no_bg = uarray(nominal_values=approx_peak, std_devs=np.sqrt(approx_std_peak**2+std_devs(approx_bg)))
        # fix into -
        peak_no_bg = xr.DataArray(data=peak_no_bg, coords=self.peak.coords)

        return peak_no_bg
