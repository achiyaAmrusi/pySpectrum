import xarray as xr
import numpy as np
import math
from uncertainties import nominal_value, std_dev, ufloat
from uncertainties.unumpy import nominal_values, std_devs
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

        properties = self.fitting_method(spectrum_slice, fitting_data)
        return properties


class GaussianWithBGFitting(PeakFit):

    def __init__(self):
        super().__init__(self.gaussian_fitting, self.plot_gaussian_fit)

    @staticmethod
    def gaussian_fitting(spectrum_slice: xr.DataArray, p0):
        """
        Fit a sum of gaussians to an xarray pyspectrum.
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

        initial_height_difference = p0[0]
        initial_baseline = p0[1]

        number_of_peaks = 0

        previous_fitting_parameters = None
        fitting_parameters = None
        finished_fitting = False

        # each iteration tries to fit another gaussian
        while number_of_peaks<=5 and (not finished_fitting):

            number_of_peaks = number_of_peaks + 1

            # Initial guess for fit parameters
            estimated_paramters = GaussianWithBGFitting.gaussian_initial_guess_estimator(spectrum_slice,
                                                                                         fitting_parameters,
                                                                                         number_of_peaks)

            initial_guess = {}
            bounds = {}
            amplitude_sum = 0
            background_center = 0
            background_fwhm = 0
            param_names = []
            for i in range(number_of_peaks):
                # initial parameters
                initial_guess.update({f'amplitude_{i}': estimated_paramters[f'amplitude_{i}'],
                                 f'mean_{i}': estimated_paramters[f'mean_{i}'],
                                 f'fwhm_{i}': estimated_paramters[f'fwhm_{i}']})
                # bounds
                bounds.update({f'amplitude_{i}': (0, np.inf),
                                 f'mean_{i}': (spectrum_slice.coords['channel'][0].item(), spectrum_slice.coords['channel'][-1].item()),
                                 f'fwhm_{i}': (0, spectrum_slice.coords['channel'][-1].item()-spectrum_slice.coords['channel'][0].item())})
                # parameters names
                param_names.extend([f'amplitude_{i}',  f'mean_{i}', f'fwhm_{i}'])
            # initial parameters
            initial_guess.update({'height_difference': nominal_value(initial_height_difference),
                                  'baseline': nominal_value(initial_baseline)})
            # bounds
            bounds.update({'height_difference': (-spectrum_slice.max().item(), spectrum_slice.max().item()),
                           'baseline': (0, spectrum_slice.max().item())})
            # parameters names
            param_names.extend(['height_difference', 'baseline'])
            # estimate the std deviation of the data
            spectrum_std_dev = GaussianWithBGFitting.estimate_error(spectrum_slice,
                                                                    initial_guess['fwhm_0'],
                                                                    initial_height_difference,
                                                                    initial_baseline,
                                                                    initial_guess['mean_0'])

            try:
                previous_fitting_parameters = fitting_parameters
                fitting_parameters = spectrum_slice.curvefit('channel', GaussianWithBGFitting.peaks,
                                                        p0=initial_guess, bounds=bounds, param_names=param_names,
                                                        kwargs={'sigma': spectrum_std_dev + 1,
                                                                'max_nfev': 50*len(param_names) })

            except:
                # if the fit didn't succeed
                if number_of_peaks<=10:
                    fitting_parameters = None
                    pass
                else:
                    return False
            if fitting_parameters is not None:
                finished_fitting = GaussianWithBGFitting.redndent_gaussian(fitting_parameters, number_of_peaks)

            if spectrum_slice.coords['channel'][0]< 295 <spectrum_slice.coords['channel'][-1]:
                print(previous_fitting_parameters, fitting_parameters)

        return previous_fitting_parameters if previous_fitting_parameters is not None else False

    @staticmethod
    def gaussian_initial_guess_estimator(spectrum_slice: xr.DataArray, fitting_parameters, number_of_peaks):
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
        estimated_paramters = {}

        if fitting_parameters is None:
            maximal_count = spectrum_slice.max().item()
            # Calculate the half-maximum count
            half_max_count = maximal_count / 2
            # Find the energy values at which the counts are closest to half-maximum on each side
            minimal_channel = spectrum_slice.where(spectrum_slice >= half_max_count, drop=True)['channel'].to_numpy()[0]
            maximal_channel = spectrum_slice.where(spectrum_slice >= half_max_count, drop=True)['channel'].to_numpy()[-1]
            # define the full width half maximum area (meaning the area which is bounded by the fwhm edges)
            fwhm_slice = spectrum_slice.sel(channel=slice(minimal_channel, maximal_channel))
            # return the mean energy in the fwhm which is the energy center

            mean = ((fwhm_slice * fwhm_slice.coords['channel']).sum() / fwhm_slice.sum()).item()
            fwhm = (maximal_channel - minimal_channel).item()
            for i in range(number_of_peaks):
                estimated_paramters.update({f'amplitude_{i}': maximal_count,
                                           f'mean_{i}': mean,
                                           f'fwhm_{i}': fwhm})
        else:
            # get the peaks from the fitting and subtract them from the previous fitting
            params = []
            estimated_paramters = {}
            for i in range(number_of_peaks-1):
                params.extend([fitting_parameters['curvefit_coefficients'].sel(param=f'amplitude_{i}').item(),
                               fitting_parameters['curvefit_coefficients'].sel(param=f'mean_{i}').item(),
                               fitting_parameters['curvefit_coefficients'].sel(param=f'fwhm_{i}').item()])
                estimated_paramters.update({f'amplitude_{i}': params[3*i],
                                            f'mean_{i}': params[3*i+1],
                                            f'fwhm_{i}': params[3 * i + 2]})
            params.extend([fitting_parameters['curvefit_coefficients'].sel(param=f'height_difference').item(),
                           fitting_parameters['curvefit_coefficients'].sel(param=f'baseline').item()])

            peaks = GaussianWithBGFitting.peaks(spectrum_slice.coords['channel'], *params)
            subtracted_spectrum = spectrum_slice - peaks
            maximal_count = subtracted_spectrum.max()
            maximal_count = maximal_count if maximal_count>0 else spectrum_slice.max()
            maximal_count_location = subtracted_spectrum.argmax() if maximal_count>0 else spectrum_slice.argmax()

            estimated_paramters.update({f'amplitude_{number_of_peaks-1}': maximal_count.item(),
                                        f'mean_{number_of_peaks-1}': subtracted_spectrum.coords['channel'][maximal_count_location.item()],
                                        f'fwhm_{number_of_peaks-1}': estimated_paramters['fwhm_0']})
        return estimated_paramters


    @staticmethod
    def peaks(domain, *params):
        """
        construct the peaks from the parameters

        """
        peaks = np.zeros_like(domain)

        num_of_parms = len(params)
        num_of_peaks = (num_of_parms-2)//3

        height_difference = params[-2]
        baseline = params[-1]

        center = 0
        fwhm = 0
        amplitude_sum = 0
        for i in range(num_of_peaks):
            peaks = peaks + GaussianWithBGFitting.gaussian(domain, params[3*i], params[3*i+1], params[3*i+2])
            center = center + params[3*i] * params[3*i+1]
            fwhm = fwhm + params[3*i] * params[3*i+2]
            amplitude_sum = amplitude_sum + params[3*i]

        center = center/amplitude_sum
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
        bg_height_difference: float
        the height difference between the right and lef edges of the peak
        bg_baseline: float
        the baseline of the peaks
        bg_center: float

        Returns
        -------
        xr.DataSet
            background values.
        """

        std = (fwhm / (2 * np.sqrt(2 * np.log(2))))
        erf_background = np.array([(math.erf(-((x - center) / (np.sqrt(2)*std))) + 1) for x in domain])
        return 0.5 * height_difference * erf_background + baseline

    @staticmethod
    def redndent_gaussian(fit_result, number_of_peaks):
        flag = False
        for i in range(number_of_peaks):
            amplitude = fit_result['curvefit_coefficients'][3*i].item()
            amplitud_error = (fit_result['curvefit_covariance'][3*i,3*i].item())**0.5
            if amplitud_error/amplitude>1:
                flag = True
                break
        return flag

    @staticmethod
    def estimate_error(spectrum_slice: xr.DataArray, fwhm , height_difference, baseline, center):

        backgrond = GaussianWithBGFitting.background(spectrum_slice.coords['channel'],
                                                                           fwhm,
                                                                           center,
                                                                           nominal_values(height_difference),
                                                                           nominal_values(baseline))
        approx_gaussian_variance = np.abs(spectrum_slice - backgrond)
        approx_variance_bg = std_devs(backgrond) ** 2
        return (approx_gaussian_variance + approx_variance_bg)**0.5

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
        peak = GaussianWithBGFitting.peaks(domain, *fit_properties['curvefit_coefficients'].values)
        plt.plot(domain, peak, color='red')

