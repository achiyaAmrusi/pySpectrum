import numpy as np


def germanium_fwhm(energy, a, b, c):
    """
    fwhm calibration function of the form of
    E(channel) = a + b*np.sqrt(energy + abs(c) * energy**2)
    """
    return a + b*np.sqrt(np.abs(energy) + np.abs(c) * energy**2)


def standard_fwhm_generator(parm):
    """
    generating a callable fwhm calibration of the form of germanium_fwhm
    Note, the function return callable
    """
    def fwhm_function(energy):
        return germanium_fwhm(energy, parm[0], parm[1], parm[2])
    return fwhm_function
