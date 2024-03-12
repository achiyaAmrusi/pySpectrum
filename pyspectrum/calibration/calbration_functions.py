import numpy as np


def germanium_fwhm(energy, a, b, c):
    return a + b*np.sqrt(energy + abs(c) * energy**2)


def standard_fwhm_generator(parm):
    def fwhm_function(energy):
        return parm[0] + parm[1]*np.sqrt(energy + parm[2] * energy**2)
    return fwhm_function
