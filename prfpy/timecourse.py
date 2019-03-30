import numpy as np
import scipy.signal as signal

def convolve_stimulus_dm(stimulus, hrf):
    """convolve_stimulus_dm
    
    convolve_stimulus_dm convolves an N-D (N>=2) stimulus array with an hrf
    
    Parameters
    ----------
    stimulus : numpy.ndarray, N-D (N>=2) 
        stimulus experimental design, with the final dimension being time
    hrf : numpy.ndarray, 1D
        contains kernel for convolution
    
    """
    hrf_shape = np.ones(len(stimulus.shape))
    hrf_shape[-1] = hrf.shape[0]
    conv_stimulus_dm = signal.fftconvolve(stimulus, hrf.reshape(hrf_shape), mode='full', axes=(-1))