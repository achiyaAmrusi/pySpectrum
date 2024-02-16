import numpy as np


def gaussian_2_dev(x, mean, fwhm):
    """
    First derivative of a Gaussian.

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
    Returns
    -------
    numpy array
        first derivaive of a Gaussian.

    """
    std = (fwhm/2.35482)
    return ((std**2-(x-mean)**2) / std**4) * 1/((2*np.pi)**0.5 * std)* np.exp(-(1/2)*((x-mean) / std)**2)
