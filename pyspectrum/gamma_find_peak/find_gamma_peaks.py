import numpy as np
import matplotlib.pyplot as plt
from pyspectrum import Spectrum, Peak
from scipy.signal import find_peaks
from .utils import gaussian_2_dev
from uncertainties import ufloat


class FindPeaks:
    """
        Find peaks in gamma spectroscopy for given statistical and resolution tolerance
        FindPeaks take a spectrum, and using the calibration can find the gamma and xray peaks in the spectrum
        The search algorithm use scipy.find_peaks with the fwhm calibration and tolerances as an input.
        The search yields estimated peak centers and fwhm under the key width (need to change it to fwhm).
        Also, PeakFinder can, given peak centers, return a list Peak object of all the peaks center given.
        The peaks domain is determined using convolution algorithm given in paper[ need to add]

        Attributes:
        - spectrum(Spectrum): The spectrum that FindPeaks search and return peaks in.
        - kernel_convolution(np.array): The convolution of the spectrum with the kernel according to []
        This kernel_convolution is calculated only once due to the time intensive cost of calculating it
           Methods:

    - `__init__(self, xarray.DataArray)`:
      Constructor method to initialize a FindPeaks instance.

    - `find_all_peaks(self, stat_error=0.05, fwhm_tol_min=0.5)`:
      find peaks in the gamma spectrum given the acceptable statistical error
      and the fwhm tolerance from the given fwhm function in spectrum(Spectrum.fwhm_calibration)

    - `plot_all_peaks(self, stat_error=0.05, fwhm_tol_min=0.5)`:
      find the peaks using find_all_peaks and plot them

    - `peak(self, peak_center:list, channel_mode=False)`:
      for a given peak_center list the function returns all the Peak object of the peaks around the peak_centers
      The domain of the peak is calculated using convolution method
    """

    def __init__(self, spectrum: Spectrum):
        """Initialize FindPeaks with a given spectrum."""
        self.spectrum = spectrum
        self.kernel_convolution = None

    def kernel(self, channel):
        """
        Generate the kernel function around the given center value(channel)
        the kernel width is defined in spectrum.fwhm_calibration.
        The kernel function here is the second derivative of gaussian
        Parameters:
        - channel: integer
            the channel in spectrum around which to make the kernel
        Returns:
        numpy array
            the values of the kernel on the domain of the spectrum channels
        """
        amplitude = 1
        energy = self.spectrum.energy_calibration(channel)
        fwhm = self.spectrum.fwhm_calibration(energy) / self.spectrum.energy_calibration[1]
        return gaussian_2_dev(self.spectrum.channels, amplitude, channel, fwhm)

    def convolution(self):
        """Convolve the spectrum with FindPeaks.kernel.
        Parameters:

        Returns:
        numpy array
            The convolution of the spectrum with thew kernel """
        convolution = []
        for i, channel in enumerate(self.spectrum.channels):
            kern_vector = self.kernel(channel)
            convolution.append(np.dot(kern_vector, self.spectrum.counts))
        return np.array(convolution)

    def peak_domain(self, peak_center, channel_mode=False):
        """ find a peak domain using the convoluted spectrum with the kernel
        Note this function works specifically for the second derivative of gaussian
        Parameters:
        - peak_center: float/integer
            estimated peak center or channel in channel mode
        - channel_model: bool
            to activate channel mode or note
        Returns:
        list
            list of touples of the domains
        """
        channels = self.spectrum.channels
        # if the convolution is not calculated yet, calculated now (time expensive procedure)
        if self.kernel_convolution is None:
            self.kernel_convolution = self.convolution()

        index_peak_center = np.where(channels > peak_center)[0][0] if channel_mode else (
            np.where(self.spectrum.xr_spectrum()['energy'] > peak_center))[0][0]
        # convolution derivative has 0 value in the peak edges
        conv_dv = self.kernel_convolution[1:] - self.kernel_convolution[:-1]
        # the behavior of the convoluted function derivative is -
        # ...minus,minus,0,plus, plus,...plus,plus,center(0 value),minus,minus,...,minus,minus,0,plus,plus...
        # the rest of this function use this to find the zero crossing in which
        # find the right edge
        flag1 = True
        flag2 = True
        right_side = conv_dv[index_peak_center:]
        index = -1
        while flag2:
            index = index + 1
            if right_side[index] < 0:
                flag2 = False
                while flag1:
                    index = index + 1
                    if right_side[index] > 0:
                        flag1 = False
        right_boundary = index_peak_center + index
        # find the left edge
        flag1 = True
        flag2 = True
        left_side = conv_dv[:index_peak_center]
        # In the right side, we start one index away from the center
        index = -1
        while flag2:
            index = index - 1
            if left_side[index] > 0:
                flag2 = False
                while flag1:
                    index = index - 1
                    if left_side[index] < 0:
                        flag1 = False

        left_boundary = index_peak_center + index
        # note that because of the derivative of the convolution the index is off by 1
        mode = self.spectrum.energy_calibration if not channel_mode else np.poly1d([1, 0])
        return mode(left_boundary + 1), mode(right_boundary + 1)

    def peak(self, peak_centers, channel_mode=False):
        """Finds the peaks which their centers are in the list.
        After finding the peak, function creating list of Peaks objects
        Parameters:
        - peak_centers - the estimated peak center (can be the one from find_all_peaks)
        - channel_mode: the default in peaks center is in energy mode. if channel_mode is true,
          peak_centers are in channels

        Returns:
        - list of Peak objects which contain the functions for spectroscopy
        """
        # this given a peak center return Peak and find all peaks gives just properties...
        background_left_value = []
        background_right_value = []
        if not (isinstance(peak_centers, list) or isinstance(peak_centers, tuple)):
            peak_center = [peak_centers]
        if channel_mode:
            peak_center = [self.spectrum.energy_calibration(peak) for peak in peak_centers]
        peak_domain = [self.peak_domain(peak) for peak in peak_centers]
        peak_fwhm = [self.spectrum.fwhm_calibration(peak) for peak in peak_centers]
        for i in range(len(peak_centers)):
            background_left = self.spectrum.xr_spectrum().sel(
                energy=slice(peak_domain[i][0] - peak_fwhm[i], peak_domain[i][0]))
            background_right = self.spectrum.xr_spectrum().sel(
                energy=slice(peak_domain[i][1], peak_domain[i][1] + peak_fwhm[i]))
            background_left_value.append(ufloat(background_left.mean(), background_left.std()))
            background_right_value.append(ufloat(background_right.mean(), background_right.std()))
        peaks = [Peak(self.spectrum.xr_spectrum().sel(energy=slice(peak_domain[i][0], peak_domain[i][1])),
                      background_left_value[i],
                      background_right_value[i]) for i in range(len(peak_centers))]
        return peaks

    def find_all_peaks(self, stat_error=0.05, fwhm_tol_min=0.5):
        """Find the peaks center and properties  in spectrum (preferably gamma spectrum) using scipy find_peaks
         and then calculate the domain of the peak using self.peak_domain.
        Note, scipy does return the domain of the peak, However, it doesn't
        seem to fit to our needs because the results are noisy and unreliable.
        Parameters:
        - stat_error - acceptable relative error in the total counts in the peak
        default: 0.05
        - fwhm_tol: tolerance of the fwhm how much can the peaks deviate from the original fwhm fit
        the deviation is only upward for cases of folding
        I don't think i actually need that in this structure, i hope to find a way to kick this factor out

        Returns:
        - tuple
        peaks_centers, properties.
        properties include the fwhm of the peaks and the left and right edges of the fwhm for the plotter

        """
        peaks_energy = []
        peaks_fwhm = []
        peaks_left_ips = []
        peaks_right_ips = []
        fwhm = self.spectrum.fwhm_calibration
        minimal_peak_area = 1 / stat_error ** 2
        # This assuming that the width is small compared with the second derivative of the calibration
        channel_fwhm = [fwhm(self.spectrum.energy_calibration(x)) /
                        self.spectrum.energy_calibration[1] for x in self.spectrum.channels]
        minimal_prominence = np.array([minimal_peak_area /
                                       (0.761438079 * np.sqrt(2 * np.pi) * (
                                               channel_fwhm[i] / (2 * np.sqrt(2 * np.log(2))))
                                        ) for i, x in enumerate(self.spectrum.channels)])
        minimal_fwhm = np.array([fwhm(x) * fwhm_tol_min for x in self.spectrum.channels])
        peaks, properties = find_peaks(self.spectrum.counts,
                                       prominence=minimal_prominence,
                                       width=minimal_fwhm,
                                       rel_height=0.5)
        peaks_energy = [self.spectrum.energy_calibration(peak) for peak in peaks]
        for i, width in enumerate(properties['widths']):
            peaks_left_ips.append(self.spectrum.energy_calibration(peaks[i] - width / 2))
            peaks_right_ips.append(self.spectrum.energy_calibration(peaks[i] + width / 2))
            peaks_fwhm.append(peaks_right_ips[-1] - peaks_left_ips[-1])

        properties['widths'] = peaks_fwhm
        properties['fwhm'] = properties.pop('widths')
        properties['fwhm_heights'] = properties.pop('width_heights')
        properties["right_ips"] = self.spectrum.energy_calibration(properties["right_ips"])
        properties["left_ips"] = self.spectrum.energy_calibration(properties["left_ips"])
        properties.pop('left_bases')
        properties.pop('right_bases')
        return peaks_energy, properties

    def plot_all_peaks(self, stat_error=0.05, fwhm_tol=0.5):
        """plot the peaks found in find_peaks
        Parameters:
        - stat_error - acceptable relative error in the total counts in the peak
        default: 0.05
        - fwhm_tol: tolerance of the fwhm how much can the peaks deviate from the original fwhm fit
        the deviation is only upward for cases of folding
        Returns:
         Nothing

    """
        peaks, properties = self.find_all_peaks(stat_error, fwhm_tol)
        # transform the peaks to index terms
        peaks_channels = [round((peak - self.spectrum.energy_calibration[0]) / self.spectrum.energy_calibration[1])
                          for peak in peaks]
        plt.semilogy(self.spectrum.energy_calibration(self.spectrum.channels), self.spectrum.counts)
        plt.semilogy(peaks, self.spectrum.counts[peaks_channels], "x")
        plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"],
                   xmax=properties["right_ips"], color="C1")
        plt.vlines(x=peaks, ymin=(self.spectrum.counts[peaks_channels] - properties["prominences"]),
                   ymax=self.spectrum.counts[peaks_channels], color="C1")
