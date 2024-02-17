from uncertainties import unumpy, ufloat, std_dev, nominal_value
import numpy as np
from scipy.optimize import curve_fit


def gaussian(x, amplitude, mean, fwhm):
    """
    Gaussian function.

    Parameters
    ----------
    x: array-like
        Input values.
    amplitude: float
        Amplitude of the Gaussian.
    mean: float
        Mean (center) of the Gaussian.
    fwhm: float
        Standard deviation/ (2 * np.sqrt(2 * np.log(2))) (width) of the Gaussian
        this is half of the distance for which the Gaussian gives half of the maximum value.

    Returns
    -------
    y: array-like
        Gaussian values.
    """
    std = (fwhm/(2 * np.sqrt(2 * np.log(2))))
    return amplitude * 1/((2*np.pi)**0.5 * std) * np.exp(-(1/2)*((x-mean) / std)**2)


def gaussian_fitting(xarray_spectrum, initial_peak_center=0, initial_fwhm=1):
    """
    Fit a Gaussian to an xarray pyspectrum.

    Parameters
    ----------
    xarray_spectrum: xarray.DataArray The pyspectrum with 'x' as the only coordinate.
    initial_peak_center: guess for initial peak center (default is 0)
    initial_std: guess for initial std (default is 1 as approximated to HPGe detectors)
    Returns
    -------
    fit_params: tuple
        The tuple containing the fit parameters (amplitude, mean, stddev) and the covariance matrix.
    """
    # Extract counts and energy values from xarray
    counts = xarray_spectrum.values

    energy_values = xarray_spectrum.coords['x'].values
    # Initial guess for fit parameters
    initial_guess = [nominal_value(np.max(counts)), initial_peak_center, initial_fwhm]

    # Perform the fit
    if (isinstance(counts[0], type(ufloat(0, 0))) or
            isinstance(counts[0], type(ufloat(0, 0)+1))):
        sigma = [std_dev(count) for count in counts]
        counts = [nominal_value(count) for count in counts]
        fit_params, cov = curve_fit(gaussian, energy_values, counts, p0=initial_guess, sigma=sigma)
    else:
        fit_params, cov = curve_fit(gaussian, energy_values, counts, p0=initial_guess)

    return fit_params, cov
