import numpy as np


def gaussian_2_dev(x, mean, fwhm):
    """
    First derivative of a Gaussian.

    Parameters
    ----------
    x: array-like
        Input values.
    mean: float
        Mean (center) of the Gaussian.
    fwhm: float
        Standard deviation/2.35482 (width) of the Gaussian
        this is half of the distance for which the Gaussian gives half of the maximum value.
    Returns
    -------
    numpy array
        second derivative of a Gaussian.

    """
    std = (fwhm/2.35482)
    return ((std**2-(x-mean)**2) / std**4) * 1/((2*np.pi)**0.5 * std) * np.exp(-(1/2)*((x-mean) / std)**2)


def asymmetrical_rect_zero_area(x, mean, width):
    """
    non-symmetrical zero area function for non-symmetrical peak recognition

    Parameters
    ----------
    x: array-like
        Input values.
    mean: float
        Mean (center) of the Gaussian.
    width: float
        width of the rect
    Returns
    -------
    numpy array

1   1-----1         1
1   1     1---------1
1   1
1----
    """
    left_right_width_factor = 0.6
    func_val = []
    domains = [mean - width/2-round(left_right_width_factor*2*width),
               mean - width / 2,
               mean + width / 2,
               mean + width / 2 + round((1-left_right_width_factor)*2*width)
               ]
    height_center = 2
    height_left = -(domains[2]-domains[1])/(domains[1]-domains[0])
    height_right = -(domains[2]-domains[1])/(domains[3]-domains[2])
    for domain_val in x:
        # if it's from the left
        if domain_val <= domains[0]:
            func_val.append(0)
        # if it's from the right
        elif domain_val > domains[3]:
            func_val.append(0)
        elif domains[0] < domain_val <= domains[1]:
            func_val.append(height_left)
        elif domains[1] < domain_val <= domains[2]:
            func_val.append(height_center)
        # last option is center_width_domain[1] <= domain_val <= right_width_domain
        else:
            func_val.append(height_right)
    return np.array(func_val)
