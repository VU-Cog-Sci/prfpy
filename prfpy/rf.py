import numpy as np
import scipy.stats as stats



def gauss1D_cart(x, mu=0.0, sigma=1.0):
    """gauss1D_cart

    gauss1D_cart takes a 1D array x, a mean and standard deviation,
    and produces a gaussian with given parameters, with a peak of height 1.

    Parameters
    ----------
    x : numpy.ndarray (1D)
        space on which to calculate the gauss
    mu : float, optional
        mean/mode of gaussian (the default is 0.0)
    sigma : float, optional
        standard deviation of gaussian (the default is 1.0)

    Returns
    -------
    numpy.ndarray
        gaussian values at x
    """

    return np.exp(-((x-mu)**2)/(2*sigma**2))



def gauss1D_log(x, mu=0.0, sigma=1.0):
    """gauss1D_log

    gauss1D_log takes a 1D array x, a mean and standard deviation,
    and produces a pRF with given parameters with the distance between mean and x log-scaled 

    Parameters
    ----------
    x : numpy.ndarray (1D)
        space on which to calculate the gauss
    mu : float, optional
        mean/mode of gaussian (the default is 0.0)
    sigma : float, optional
        standard deviation of gaussian (the default is 1.0)

    Returns
    -------
    numpy.ndarray
        gaussian values at log(x)
    """

    return np.exp(-(np.log(x/mu)**2)/(2*sigma**2))



def vonMises1D(x, mu=0.0, kappa=1.0):
    """vonMises1D

    vonMises1D takes a 1D array x, a mean and kappa (inverse of standard deviation),
    and produces a von Mises pRF with given parameters. This shape can be thought of 
    as a circular gaussian shape. Used for orientation or motion direction pRFs, 
    for instance.

    Parameters
    ----------
    x : numpy.ndarray (1D)
        space on which to calculate the von Mises. 
        Assumed to be in the range (0, 2*np.pi)
    mu : float, optional
        mean/mode of von Mises (the default is 0.0)
    kappa : float, optional
        dispersion coefficient of the von Mises, 
        akin to invers of standard deviation of gaussian (the default is 1.0)

    Returns
    -------
    numpy.ndarray
        von Mises values at x, peak has y-value of 1
    """
    vm = stats.vonmises.pdf(x-mu, kappa)

    return vm / np.max(vm)



def gauss2D_iso_cart(x, y, mu=(0.0, 0.0), sigma=1.0, normalize_RFs=False):
    """gauss2D_iso_cart

    gauss2D_iso_cart takes two-dimensional arrays x and y, containing
    the x and y coordinates at which to evaluate the 2D isotropic gaussian 
    function, with a given sigma, and returns a 2D array of Z values.

    Parameters
    ----------
    x : numpy.ndarray, 2D or flattened by masking
        2D, containing x coordinates
    y : numpy.ndarray, 2D or flattened by masking
        2D, containing y coordinates
    mu : tuple, optional
        mean, 2D coordinates of mean/mode of gauss (the default is (0.0,0.0))
    sigma : float, optional
        standard deviation of gauss (the default is 1.0)

    Returns 
    -------
    numpy.ndarray, 2D or flattened by masking
        gaussian values evaluated at (x,y)
    """
    if normalize_RFs:
        return np.exp(-((x-mu[0])**2 + (y-mu[1])**2)/(2*sigma**2)) /(2*np.pi*sigma**2)
    else:
        return np.exp(-((x-mu[0])**2 + (y-mu[1])**2)/(2*sigma**2))



def gauss2D_rot_cart(x, y, mu=(0.0, 0.0), sigma=1.0, theta=0.0, ar=1.0):
    """gauss2D_rot_cart

    gauss2D_rot_cart takes two-dimensional arrays x and y, containing
    the x and y coordinates at which to evaluate the 2D non-isotropic gaussian 
    function, with a given sigma, angle of rotation theta, and aspect ratio ar.
    it returns a 2D array of Z values. Default is an isotropic gauss.

    Parameters
    ----------
    x : numpy.ndarray, 2D
        2D, containing x coordinates or flattened by masking
    y : numpy.ndarray, 2D
        2D, containing y coordinates or flattened by masking
    mu : tuple, optional
        mean, 2D coordinates of mean/mode of gauss (the default is (0.0,0.0))
    sigma : float, optional
        standard deviation of gauss (the default is 1.0)
    theta : float, optional
        angle of rotation of gauss (the default is 0.0)   
    ar : float, optional
        aspect ratio of gauss, multiplies the rotated y parameters (the default is 1.0)

    Returns
    -------
    numpy.ndarray, 2D or flattened by masking
        gaussian values evaluated at (x,y) 
    """
    xr = (x-mu[0]) * np.cos(theta) + (y-mu[1]) * np.sin(theta)
    yr = -(x-mu[0]) * np.sin(theta) + (y-mu[1]) * np.cos(theta)

    return np.exp(-(xr**2 + ar**2 * yr**2)/(2*sigma**2))



def gauss2D_logpolar(ecc, polar, mu=(1.0, 0.0), sigma=1.0, kappa=1.0):
    """gauss2D_logpolar

    gauss2D_logpolar takes two-dimensional arrays ecc and polar, containing
    the eccentricity and polar angle coordinates at which to evaluate the 2D gaussian, 
    which in this case is a von Mises in the polar angle direction, and a log gauss 
    in the eccentricity dimension, and returns a 2D array of Z values.
    We recommend entering the ecc and polar angles ordered as x and y for easy
    visualization.

    Parameters
    ----------
    ecc : numpy.ndarray, 2D or flattened by masking
        2D, containing eccentricity
    polar : numpy.ndarray, 2D or flattened by masking
        2D, containing polar angle coordinates (0, 2*np.pi)
    mu : tuple, optional
        mean, 2D coordinates of mean/mode of gauss (ecc) and von Mises (polar) (the default is (0.0,0.0))
    sigma : float, optional
        standard deviation of gauss (the default is 1.0)
    kappa : float, optional
        dispersion coefficient of the von Mises, 
        akin to inverse of standard deviation of gaussian (the default is 1.0)

    Returns
    -------
    numpy.ndarray, 2D or flattened by masking
        values evaluated at (ecc, polar), peak has y-value of 1.
    """
    ecc_gauss = np.exp(-(np.log(ecc/mu[0])**2)/(2*sigma**2))
    polar_von_mises = stats.vonmises.pdf(polar-mu[1], kappa)
    polar_von_mises /= np.max(polar_von_mises)
    logpolar_Z = ecc_gauss * polar_von_mises

    return logpolar_Z / np.max(logpolar_Z)
