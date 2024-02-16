import numpy as np

ge_standard_fwhm_calibration_coeff = [0, 0.02]


def germanium_fwhm(energy, a, b):
    return np.sqrt(a+b*energy)
