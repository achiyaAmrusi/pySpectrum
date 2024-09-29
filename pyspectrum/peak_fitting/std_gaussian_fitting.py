import warnings
import math
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from uncertainties import nominal_value, std_dev
from pyspectrum.peak_fitting.abstract_fitting_class import PeakFit


class GaussianWithBGFitting(PeakFit):

    def __init__(self):
        """
        For now no initialization is needed
         """

    @property
    def fit_type(self):
        return "Gaussian with erf background"

    @staticmethod
    def fit(spectrum_slice: xr.DataArray, **kwargs):
        """
        Fit a Gaussian to xarray pyspectrum.
        If the fit fails return False
        Parameters
        ----------
        spectrum_slice: xarray.DataArray
          spectrum slice of the peak with one coordinate
        kwargs:
            peaks_centers: np.ndarray
                an array of the peaks centers
            estimated_fwhm: float
                the approximated fwhm in the spectrum slice
            background_parameters: list of 2 ufloat
                parameters for the fit = [height_difference, peak_baseline] of ufloat
         Returns:
         --------
         - fit_params: tuple
             The tuple containing the fit parameters (amplitude, mean, stddev) and the covariance matrix.
         """

        try:
            peaks_centers = kwargs.get('peaks_centers')
            estimated_fwhm = kwargs.get('estimated_fwhm')
            background_parameters = kwargs.get('background_parameters')
        except ValueError:
            raise ValueError

        # define the coordinate name
        coord_name = list(spectrum_slice.coords.keys())[0]
        spectrum_slice = spectrum_slice.rename({spectrum_slice.coords[coord_name].name: 'channel'})

        # Initial guess for fit parameters
        initial_guess, param_names = GaussianWithBGFitting.create_initial_guess(spectrum_slice, peaks_centers,
                                                                                estimated_fwhm, background_parameters)
        bounds = GaussianWithBGFitting.create_bounds(spectrum_slice, len(peaks_centers))

        # calculate the std of the data for sigma in curvefit
        std = (estimated_fwhm / (2 * np.sqrt(2 * np.log(2))))
        mean = (spectrum_slice * spectrum_slice.channel).sum() / spectrum_slice.sum()
        # approximate the gaussian part and the background part of the peak
        erf = np.array(
            [(math.erf(-(x - mean) / std) + 1) for x in spectrum_slice.coords['channel'].to_numpy()])
        approx_nominal_bg = 0.5 * initial_guess['height_difference'] * erf + initial_guess['peak_baseline']
        approx_nominal_gauss = spectrum_slice - approx_nominal_bg
        approx_var_bg = ((0.5 * std_dev(background_parameters[0]) * erf) ** 2 + std_dev(background_parameters[1]) ** 2)
        approx_var_gauss = abs(approx_nominal_gauss)

        try:
            fit_result = spectrum_slice.curvefit('channel', GaussianWithBGFitting.peaks,
                                                 p0=initial_guess, bounds=bounds, param_names=param_names,
                                                 kwargs={'sigma': (approx_var_gauss + approx_var_bg) ** 0.5 + 1})
        except ValueError as e:
            # if the fit didn't succeed
            warnings.warn(f"The fit of domain [{spectrum_slice.channel.values[0]}, {spectrum_slice.channel.values[-1]}]"
                          f"have failed due to: {ValueError}\n")
            return False

        return fit_result

    @staticmethod
    def plot_fit(domain, fit_properties):
        """
        Plot the peak in the domain given

        Parameters
        ----------
        domain: array-like
        the domain on which the peak is defined
        fit_properties: xr.DataSet
            Gaussian plus background values.
            """
        peak = GaussianWithBGFitting.peaks(domain, *fit_properties['curvefit_coefficients'].values)
        plt.plot(domain, peak, color='red')

    @staticmethod
    def create_initial_guess(spectrum_slice: xr.DataArray, peaks_centers: np.ndarray, fwhm: float,
                             background_parameters: list):
        """
````    `creates initial guess for the fit using xarray.
        Parameters
        ----------
        spectrum_slice: xarray.DataArray
            spectrum slice of the peak with one coordinate
        peaks_centers: np.ndarray
            an array of the peak centers
        fwhm: float
            the value the resolution
        background_parameters: list
            parameters for the fit = [height_difference, peak_baseline] of ufloat
        Returns
        -------
        - Dict
            initial guess for xarray
        """
        peak_number = len(peaks_centers)
        initial_guess = {}
        param_names = []
        for i in range(peak_number):
            initial_guess.update({f'amplitude_{i}': spectrum_slice.sel(channel=peaks_centers[i]).values,
                                  f'mean_{i}': peaks_centers[i],
                                  f'fwhm_{i}': fwhm})
            param_names.extend([f'amplitude_{i}', f'mean_{i}', f'fwhm_{i}'])
        initial_guess.update({
            'height_difference': np.max([nominal_value(background_parameters[0]), 1]),
            'peak_baseline': np.max([nominal_value(background_parameters[1]), 0])
        })
        param_names.extend(['height_difference', 'baseline'])
        return initial_guess, param_names

    @staticmethod
    def create_bounds(spectrum_slice: xr.DataArray, number_of_peaks: np.ndarray):
        """
````    `creates initial guess for the fit using xarray.
        Parameters
        ----------
        spectrum_slice: xarray.DataArray
            spectrum slice of the peak with one coordinate
        number_of_peaks: np.ndarray
            an array of the peak centers
        Returns
        -------
        - Dict
            bounds for xarray
        """
        bounds = {}

        for i in range(number_of_peaks):
            bounds.update({f'amplitude_{i}': (1, np.inf),
                           f'mean_{i}': (spectrum_slice.channel.values[0], spectrum_slice.channel.values[-1]),
                           f'fwhm_{i}': (0, spectrum_slice.channel.values[-1] - spectrum_slice.channel.values[0])})

        bounds.update({
            'height_difference': (0, np.inf),
            # need better idea for the peak baseline
            'peak_baseline': (0, np.inf)
        })
        return bounds

    @staticmethod
    def peaks(domain, *params):
        """
        construct the peaks from the parameters
        """
        peaks = np.zeros_like(domain)
        num_of_parms = len(params)
        num_of_peaks = (num_of_parms - 2) // 3

        height_difference = params[-2]
        baseline = params[-1]

        center = 0
        fwhm = 0
        amplitude_sum = 0
        for i in range(num_of_peaks):
            peaks = peaks + GaussianWithBGFitting.gaussian(domain, params[3 * i], params[3 * i + 1], params[3 * i + 2])
            center = center + params[3 * i] * params[3 * i + 1]
            fwhm = fwhm + params[3 * i] * params[3 * i + 2]
            amplitude_sum = amplitude_sum + params[3 * i]

        center = center / amplitude_sum
        fwhm = fwhm / amplitude_sum
        peaks = peaks + GaussianWithBGFitting.background(domain, fwhm, center, height_difference, baseline)

        return xr.DataArray(peaks, coords={'coords': domain})

    @staticmethod
    def gaussian(domain, amplitude, mean, fwhm):
        """
        Gaussian function.

        Parameters
        ----------
        domain: array-like
         domain on which the gaussian is defined
        amplitude: float
         Amplitude of the Gaussian.
        mean: float
         Mean (center) of the Gaussian.
        fwhm: float
         Standard deviation/ (2 * np.sqrt(2 * np.log(2))) (width) of the Gaussian
         this is half of the distance for which the Gaussian gives half of the maximum value.

        Returns
        -------
        xr.DataSet
            Gaussian plus background values.
        """

        std = (fwhm / (2 * np.sqrt(2 * np.log(2))))
        gaussian = amplitude * 1 / ((2 * np.pi) ** 0.5 * std) * np.exp(-(1 / 2) * ((domain - mean) / std) ** 2)
        return gaussian

    @staticmethod
    def background(domain, fwhm, center, height_difference, baseline):
        """
        background function.

        Parameters
        ----------
        domain: array-like
            domain on which the gaussian is defined
        fwhm: float
            Standard deviation/ (2 * np.sqrt(2 * np.log(2))) (width) of the Gaussian
            this is half of the distance for which the Gaussian gives half of the maximum value.
        center: float
            The center of the background erf
        height_difference: float
            the height difference between the right and lef edges of the peak
        baseline: float
            The baseline height
        Returns
        -------
        xr.DataSet
            background values.
        """

        std = (fwhm / (2 * np.sqrt(2 * np.log(2))))
        erf_background = np.array([(math.erf(-((x - center) / (np.sqrt(2) * std))) + 1) for x in domain])
        return 0.5 * height_difference * erf_background + baseline
