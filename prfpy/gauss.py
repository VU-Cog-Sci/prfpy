import numpy as np

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

    return np.exp(-(np.log(x-mu)**2)/(2*sigma**2))

def gauss2D_iso_cart(x, y, mu=(0.0,0.0), sigma=1.0):
    """gauss2D_iso_cart
    
    gauss2D_iso_cart takes two-dimensional arrays x and y, containing
    the x and y coordinates at which to evaluate the 2D isotropic gaussian 
    function, with a given sigma, and returns a 2D array of Z values.
        
    Parameters
    ----------
    x : numpy.ndarray, 2D
        2D, containing x coordinates
    y : numpy.ndarray, 2D
        2D, containing y coordinates
    mu : tuple, optional
        mean, 2D coordinates of mean/mode of gauss (the default is (0.0,0.0))
    sigma : float, optional
        standard deviation of gauss (the default is 1.0
    
    Returns
    -------
    numpy.ndarray, 2D
        gaussian values evaluated at (x,y)
    """
    
    return np.exp(-((x-mu[0])**2 + ar**2 * (x-mu[1])**2)/(2*sigma**2))

def gauss2D_rot_cart(x, y, mu=(0.0,0.0), sigma=1.0, theta=0.0, ar=1.0):
    """gauss2D_rot_cart
    
    gauss2D_rot_cart takes two-dimensional arrays x and y, containing
    the x and y coordinates at which to evaluate the 2D non-isotropic gaussian 
    function, with a given sigma, angle of rotation theta, and aspect ratio ar.
    it returns a 2D array of Z values. Default is an isotropic gauss.
        
    Parameters
    ----------
    x : numpy.ndarray, 2D
        2D, containing x coordinates
    y : numpy.ndarray, 2D
        2D, containing y coordinates
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
    numpy.ndarray, 2D
        gaussian values evaluated at (x,y)
    """
    xr = (x-mu[0]) * np.cos(theta) + (y-mu[1]) * np.sin(theta)
    yr = -(x-mu[0]) * np.sin(theta) + (y-mu[1]) * np.cos(theta)
    return np.exp(-(xr**2 + ar**2 * yr**2)/(2*sigma**2))

