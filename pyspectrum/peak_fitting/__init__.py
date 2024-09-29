""" A module for fitting objects. \n The fitting objects inherit from PeakFit"""
from .abstract_fitting_class import PeakFit
from .std_gaussian_fitting import GaussianWithBGFitting
