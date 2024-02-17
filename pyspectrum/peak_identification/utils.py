import numpy as np


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
        Standard deviation/2.35482 (width) of the Gaussian
        this is half of the distance for which the Gaussian gives half of the maximum value.

    Returns:
    array-like
        Gaussian values in x.
    """
    return amplitude * np.exp(-(1/2)*((x-mean) / (fwhm/2.35482))**2)
