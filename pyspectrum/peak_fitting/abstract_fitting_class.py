import xarray as xr
from abc import ABC, abstractmethod


class PeakFit(ABC):
    """
    Abstract base class for peak fitting methods.
    Any peak fitting class must implement the `fit` and `plot_fit` methods.
    Must be implemented by subclasses.
    Methods
    ----------
    fit: callable
    given spectrum slice and fitting data (initial data which is required from the fitting methods) the function
    returns the fit
    plot_fit: callable
    plot the fit from the results of fitting methods
    """

    @property
    @abstractmethod
    def fit_type(self):
        """
        Abstract property for the type of fitting (e.g., Gaussian, Lorentzian).
         """
        pass

    @staticmethod
    @abstractmethod
    def fit(spectrum_slice: xr.DataArray, **kwargs):
        """
        Fit the given spectrum slice and return the fit properties.
        """
        pass

    @staticmethod
    @abstractmethod
    def plot_fit(domain, fit_properties):
        """
        Plot the results of the fit.
        """
        pass

