import numpy as np
from scipy.optimize import curve_fit
from pyspectrum import Spectrum
from .calbration_functions import germanium_fwhm, ge_standard_fwhm_calibration_coeff


class Calibration:
    """"""

    def __init__(self, spectrum:Spectrum):
        self.spectrum = spectrum
        self.peaks = []
        self.peaks_energy = []

    def add_peaks(self, peak_domain_list: list, energy_list: list):
        if not len(peak_domain_list) == len(energy_list):
            raise f"peak_domain_list and energy_list need to have the same length"
        for ind, peak_domain in enumerate(peak_domain_list):
    #        self.peaks.append(Peak(self.spectrum.xr_spectrum().sel(energy=slice(peak_domain[0],peak_domain[1]))))
            self.peaks_energy.append(energy_list[ind])

    def generate_energy_calibration(self, degree_of_poly: int):

        peaks_center = []
        for peak in self.peaks:
            peaks_center.append(peak.first_moment_method_center())
        return np.polyfit(peaks_center, self.peaks_energy, deg=degree_of_poly)

    def generate_fwhm_calibration(self, resolution_function=germanium_fwhm, p0=ge_standard_fwhm_calibration_coeff):
        """
        f : callable
        """
        peaks_fwhm = []
        for peak in self.peaks:
            peaks_fwhm.append(peak.estimated_resolution)
        return curve_fit(resolution_function, self.peaks_energy, peaks_fwhm, p0=p0)
