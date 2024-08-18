import xarray as xr
import numpy as np
import math
from uncertainties import nominal_value, std_dev
import matplotlib.pyplot as plt


class PeakFit:
    """
    The class is a template object for fitting peaks in a spectrum.
    each object that is used for fitting need to inherent from PeakFit, such that it takes the same parameters
    (maybe more)
    The class takes spectrum slice, fitting method and other plotted for the given fitting method

    Attributes
    ----------
   peak_overlap: bool
   suppose to not exist later in the development process, if true the function fit for serval peaks
   fitting_method: callable
    given spectrum slice and fitting data (initial data which is required from the fitting methods) the function
    returns the fit
    plot_fit: callable
    plot the fit from the results of fitting methods
    """

    def __init__(self, fitting_method=None, plot_fit=None):
        self.fitting_method = fitting_method
        self.plot_fit = plot_fit

    def peak_fit(self, spectrum_slice: xr.DataArray, fitting_data=None):
        """
        activate the fit of fitting methods

        Parameters
        ----------
        spectrum_slice: xr.DataArray
        the slice of the peak in the spectrum for the fit
        fitting_data: list
        data for the fitting, this is fit method dependent
        """

        num_of_peaks = 1
        properties = self.fitting_method(spectrum_slice, fitting_data)
        return properties, num_of_peaks


class GaussianWithBGFitting(PeakFit):

    def __init__(self):
        super().__init__(self.gaussian_fitting, self.plot_gaussian_fit)

    @staticmethod
    def gaussian_fitting(spectrum_slice: xr.DataArray, p0):
        """
        Fit a Gaussian to an xarray pyspectrum.
        If the fit fails return False
        Parameters
        ----------
        spectrum_slice: xarray.DataArray
          spectrum slice of the peak with one coordinate
        p0 : list
          parameters for the fit. p0 = [counts_from_the_left, counts_from_the_right]
         Returns:
         - fit_params: tuple
             The tuple containing the fit parameters (amplitude, mean, stddev) and the covariance matrix.
         """
        # define the coordinate name
        coord_name = list(spectrum_slice.coords.keys())[0]
        spectrum_slice = spectrum_slice.rename({spectrum_slice.coords[coord_name].name: 'channel'})

        # Initial guess for fit parameters
        estimated_amplitude, estimated_center, estimated_fwhm = GaussianWithBGFitting.gaussian_initial_guess_estimator(
            spectrum_slice)
        # nominal guess for curvefit
        initial_guess = {'amplitude': nominal_value(estimated_amplitude.item()),
                         'mean': nominal_value(estimated_center.item()),
                         'fwhm': estimated_fwhm,
                         'height_difference': nominal_value(p0[0]),
                         'peak_baseline': nominal_value(p0[1])}
        # calculate the std of the data for sigma in curvefit
        std = (initial_guess['fwhm'] / (2 * np.sqrt(2 * np.log(2))))
        # approximate the gaussian part and the background part of the peak
        erf = np.array(
            [(math.erf(-(x - initial_guess['mean']) / std) + 1) for x in spectrum_slice.coords['channel'].to_numpy()])
        approx_nominal_bg = 0.5 * initial_guess['height_difference'] * erf + initial_guess['peak_baseline']
        approx_nominal_gauss = spectrum_slice - approx_nominal_bg
        approx_var_bg = ((0.5 * std_dev(p0[0]) * erf) ** 2 + std_dev(p0[1]) ** 2)
        approx_var_gauss = abs(approx_nominal_gauss)
        # Perform the fit
        total_width = spectrum_slice.coords['channel'][-1] - spectrum_slice.coords['channel'][0]
        bin_size = spectrum_slice.coords['channel'][1] - spectrum_slice.coords['channel'][0]
        bounds = {'amplitude': (0, np.inf),
                  'mean': (spectrum_slice.coords['channel'][0], spectrum_slice.coords['channel'][-1]),
                  'fwhm': (bin_size, total_width),
                  'height_difference': (-np.inf, np.inf),
                  'peak_baseline': (0, np.inf)}
        try:
            fit_result = spectrum_slice.curvefit('channel', GaussianWithBGFitting.gaussian_with_bg,
                                                 p0=initial_guess, bounds=bounds,
                                                 kwargs={'sigma': (approx_var_gauss + approx_var_bg) ** 0.5 + 1})
        except:
            # if the fit didn't succeed
            return False
        # if the baseline is negative or the height difference is negative it is not a peak
        negative_baseline = fit_result['curvefit_coefficients'].sel(param='bg_baseline').item() < \
                            - fit_result['curvefit_covariance'].sel(cov_i='bg_baseline', cov_j='bg_baseline')
        negative_height_difference = fit_result['curvefit_coefficients'].sel(param='bg_height_difference').item() < \
                                     - fit_result['curvefit_covariance'].sel(cov_i='bg_height_difference',
                                                                             cov_j='bg_height_difference')
#        if negative_baseline or negative_height_difference:
#            return False
        return [fit_result]

    @staticmethod
    def gaussian_initial_guess_estimator(peak: xr.DataArray):
        """
        estimate the center of the peak
        the function operate by the following order -
        1. find maximal value
        2. define domain within fwhm edges
        3. calculate the mean energy which is the center of the peak (like center of mass)

        Parameters
        ----------
        peak: xarray.DataArray
          spectrum slice of a peak with one coordinate

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
        return (maximal_count,
                (fwhm_slice * fwhm_slice.coords['channel']).sum() / fwhm_slice.sum(),
                (maximal_channel - minimal_channel))

    @staticmethod
    def gaussian_with_bg(domain, amplitude, mean, fwhm, bg_height_difference, bg_baseline, bg_center):
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
        bg_height_difference: float
        the height difference between the right and lef edges of the peak
        bg_baseline: float
        the baseline of the peaks
        bg_center: float
        Returns
        -------
        xr.DataSet
            Gaussian plus background values.
        """

        std = (fwhm / (2 * np.sqrt(2 * np.log(2))))
        gaussian = amplitude * 1 / ((2 * np.pi) ** 0.5 * std) * np.exp(-(1 / 2) * ((domain - mean) / std) ** 2)
        erf_background = np.array([(math.erf(-((x - bg_center) / std)) + 1) for x in domain])
        background = 0.5 * bg_height_difference * erf_background + bg_baseline
        return gaussian + erf_background + background

    @staticmethod
    def plot_gaussian_fit(domain, fit_properties):
        """
        Plot the peak in the domain given

        Parameters
        ----------
        domain: array-like
        the domain on which the peak is defined
        fit_properties: xr.DataSet
            Gaussian plus background values.
            """
        mean = fit_properties['curvefit_coefficients'].sel(param='mean').item()
        amplitude = fit_properties['curvefit_coefficients'].sel(param='amplitude').item()
        fwhm = fit_properties['curvefit_coefficients'].sel(param='fwhm').item()
        height_difference = fit_properties['curvefit_coefficients'].sel(param='bg_height_difference').item()
        peak_baseline = fit_properties['curvefit_coefficients'].sel(param='bg_baseline').item()
        background_center = fit_properties['curvefit_coefficients'].sel(param='bg_center').item()
        peak = GaussianWithBGFitting.gaussian_with_bg(domain,
                                                      amplitude,
                                                      mean,
                                                      fwhm,
                                                      height_difference,
                                                      peak_baseline,
                                                      background_center)
        plt.plot(domain, peak, color='red')

