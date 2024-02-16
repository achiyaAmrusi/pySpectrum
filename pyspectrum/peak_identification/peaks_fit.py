import xarray as xr
import numpy as np
import math
from uncertainties import nominal_value, std_dev, ufloat
import matplotlib.pyplot as plt


class PeakFit:

    def __init__(self, peak_overlap: bool, fitting_method=None, plot_fit=None):
        self.peak_overlap = peak_overlap
        self.fitting_method = fitting_method
        self.plot_fit = plot_fit

    def peak_fit(self, spectrum_slice: xr.DataArray, fitting_data=None):
        """
        The class is a general  object for fitting peaks in a spectrum.
        The class takes spectrum slice, fitting method and other parameter for the fitting method
        and use it to fit the data.
        given a new peak type, the only thing needed is to create new peak fitting method, but.
         the rest of pyspectrum.peak_identification suppose to stay functioning.
        Parameters
        ----------

            - counts per channel (xarray.DataArray): counts per channel for all the peak.
            peak_fit_function -must have as parameters in the following order amp, center. fwhm so on?
            """
        if self.peak_overlap:
            num_of_peaks, properties = self.fitting_method(spectrum_slice, fitting_data)
        else:
            num_of_peaks = 1
            properties = self.fitting_method(spectrum_slice, fitting_data)
        return properties, num_of_peaks


class GaussianWithBGFitting(PeakFit):

    def __init__(self, peak_overlap):
        if peak_overlap:
            super().__init__(peak_overlap, self.gaussian_fitting_method_overlap, self.plot_gaussian_fit)
        else:
            super().__init__(peak_overlap, self.single_gaussian_fitting, self.plot_gaussian_fit)

    @staticmethod
    def gaussian_fitting_method_overlap(spectrum_slice: xr.DataArray, p0):
        """
         Fit multiple gaussian functions with background to a xarray pyspectrum.
         The method is to try and fit n gaussian functions to the spectrum,
          reduce the fitted functions from the spectrum and then check for a signal using find_peaks.
          if a signal is detected the method tries to fit n+1 gaussian functions to the peak

        Parameters
        ----------
        spectrum_slice: xarray.DataArray
          The pyspectrum with 'x' as the only coordinate.
         p0 : list
          parameters for the fit. p0 = [fwhm, counts_from_the_left, counts_from_the_right]
        Returns
        -------
         - fit_params: tuple
             The tuple containing the fit parameters (amplitude, mean, stddev) and the covariance matrix.
         """
        # fix the coordinate name
        coord_name = list(spectrum_slice.coords.keys())[0]
        spectrum_slice = spectrum_slice.rename({spectrum_slice.coords[coord_name].name: 'channel'})

        # Initial guess for fit parameters
        estimated_amplitude, estimated_center, estimated_fwhm = GaussianWithBGFitting.gaussian_initial_guess_estimator(
            spectrum_slice)
        initial_guess = {'amplitude': estimated_amplitude,
                         'mean': estimated_center,
                         'fwhm': estimated_fwhm,
                         'height_difference': nominal_value(p0[0]),
                         'peak_baseline': nominal_value(p0[1])}

        # Perform the fit
        if (isinstance(spectrum_slice.values[0], type(ufloat(0, 0))) or
                isinstance(spectrum_slice.values[0], type(ufloat(0, 0) + 1))):
            sigma = [std_dev(count) for count in spectrum_slice.values]
            fit_result = spectrum_slice.curvefit('channel', GaussianWithBGFitting.gaussian_with_bg,
                                                 p0=initial_guess, sigma=sigma)
        else:
            fit_result = spectrum_slice.curvefit('channel', GaussianWithBGFitting.gaussian_with_bg,
                                                 p0=initial_guess)
        return [fit_result]

    @staticmethod
    def single_gaussian_fitting(spectrum_slice: xr.DataArray, p0):
        """
         Fit a Gaussian to an xarray pyspectrum.

        Parameters
        ----------
        spectrum_slice: xarray.DataArray
          The pyspectrum with 'x' as the only coordinate.
        p0 : list
          parameters for the fit. p0 = [counts_from_the_left, counts_from_the_right]
         Returns:
         - fit_params: tuple
             The tuple containing the fit parameters (amplitude, mean, stddev) and the covariance matrix.
         """
        # fix the coordinate name
        coord_name = list(spectrum_slice.coords.keys())[0]
        spectrum_slice = spectrum_slice.rename({spectrum_slice.coords[coord_name].name: 'channel'})

        # Initial guess for fit parameters
        estimated_amplitude, estimated_center, estimated_fwhm = GaussianWithBGFitting.gaussian_initial_guess_estimator(
            spectrum_slice)
        initial_guess = {'amplitude': estimated_amplitude,
                         'mean': estimated_center,
                         'fwhm': estimated_fwhm,
                         'height_difference': nominal_value(p0[0]),
                         'peak_baseline': nominal_value(p0[1])}

        # Perform the fit
        if (isinstance(spectrum_slice.values[0], type(ufloat(0, 0))) or
                isinstance(spectrum_slice.values[0], type(ufloat(0, 0) + 1))):
            sigma = [std_dev(count) for count in spectrum_slice.values]
            fit_result = spectrum_slice.curvefit('channel', GaussianWithBGFitting.gaussian_with_bg,
                                                 p0=initial_guess, sigma=sigma)
        else:
            fit_result = spectrum_slice.curvefit('channel', GaussianWithBGFitting.gaussian_with_bg,
                                                 p0=initial_guess)
        return [fit_result]

    @staticmethod
    def gaussian_initial_guess_estimator(peak: xr.DataArray):
        """ calculate the center of the peak
        the function operate by the following order -
         calculate the peak domain,
          find maximal value
          define domain within fwhm edges
          calculate the mean energy which is the center of the peak (like center of mass)
        Parameters
        ----------
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
        return (maximal_count,
                (fwhm_slice * fwhm_slice.coords['channel']).sum() / fwhm_slice.sum(),
                (maximal_channel - minimal_channel))

    @staticmethod
    def gaussian_with_bg(domain, amplitude, mean, fwhm, height_difference, peak_baseline):
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
        height_difference: float
        the height difference between the right and lef edges of the peak
        peak_baseline: float
        the baseline of the peaks
        Returns
        -------
        y array-like
            Gaussian plus background values.
        """

        std = (fwhm / (2 * np.sqrt(2 * np.log(2))))
        gaussian = amplitude * 1 / ((2 * np.pi) ** 0.5 * std) * np.exp(-(1 / 2) * ((domain - mean) / std) ** 2)
        erf_background = np.array([(math.erf(-((x - mean) / std)) + 1) for x in domain])
        background = 0.5 * height_difference * erf_background + peak_baseline
        return gaussian + erf_background + background

    @staticmethod
    def plot_gaussian_fit(domain, fit_properties):
        """"given fit properties plot the peak in the domain given"""
        mean = fit_properties['curvefit_coefficients'].sel(param='mean').item()
        amplitude = fit_properties['curvefit_coefficients'].sel(param='amplitude').item()
        fwhm = fit_properties['curvefit_coefficients'].sel(param='fwhm').item()
        height_difference = fit_properties['curvefit_coefficients'].sel(param='height_difference').item()
        peak_baseline = fit_properties['curvefit_coefficients'].sel(param='peak_baseline').item()
        peak = GaussianWithBGFitting.gaussian_with_bg(domain,
                                                      amplitude,
                                                      mean,
                                                      fwhm,
                                                      height_difference,
                                                      peak_baseline)
        plt.plot(domain, peak, color='red')
