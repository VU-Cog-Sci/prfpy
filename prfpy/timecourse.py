import numpy as np
import scipy as sp
import scipy.signal as signal
from statsmodels.tsa.arima_process import arma_generate_sample


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
    hrf_shape = np.ones(len(stimulus.shape), dtype=np.int)
    hrf_shape[-1] = hrf.shape[-1]

    return signal.fftconvolve(stimulus, hrf.reshape(hrf_shape), mode='full', axes=(-1))[..., :stimulus.shape[-1]]


def stimulus_through_prf(prfs, stimulus, dx, mask=None):
    """stimulus_through_prf

    dot the stimulus and the prfs

    Parameters
    ----------
    prfs : numpy.ndarray
        the array of prfs. 
    stimulus : numpy.ndarray
        the stimulus design matrix, either convolved with hrf or not.
    mask : numpy.ndarray
        a mask in feature space, of dimensions equal to 
        the spatial dimensions of both stimulus and receptive field

    """
    assert prfs.shape[1:] == stimulus.shape[:-1], \
        """prf array dimensions {prfdim} and input stimulus array dimensions {stimdim} 
        must have same dimensions""".format(
            prfdim=prfs.shape[1:],
            stimdim=stimulus.shape[:-1])
    if mask == None:
        prf_r = prfs.reshape((prfs.shape[0], -1))
        stim_r = stimulus.reshape((-1, stimulus.shape[-1]))
    else:
        assert prfs.shape[1:] == mask.shape and mask.shape == stimulus.shape[:-1], \
            """mask dimensions {maskdim}, prf array dimensions {prfdim}, 
            and input stimulus array dimensions {stimdim} 
            must have same dimensions""".format(
                maskdim=mask.shape,
                prfdim=prfs.shape[1:],
                stimdim=stimulus.shape[:-1])
        prf_r = prfs[:, mask]
        stim_r = stimulus[mask, :]
    return prf_r @ stim_r * (dx ** len(stimulus.shape[:-1]))


def filter_predictions(predictions, 
                       filter_type,
                       filter_params):
    """
    Generic filtering function, calling the different types of filters implemented.

    Parameters
    ----------
    
    See individual filters for description.


    Returns
    -------
    numpy.ndarray
        filtered version of the array
        
    """
    
    if filter_type == 'sg':
        return sgfilter_predictions(predictions,
                                    **filter_params)
    elif filter_type == 'dc':
        return dcfilter_predictions(predictions,
                                    **filter_params)
    else:
        print("unknown filter option selected, using unfiltered prediction")
        return predictions


def dcfilter_predictions(predictions, first_modes_to_remove=5,
                         last_modes_to_remove_percent=0,
                         add_mean=True,
                         task_lengths=None,
                         task_names=None, late_iso_dict=None, **kwargs):
    """dcfilter_predictions

    discrete cosine filter predictions, to conform to data filtering

    Parameters
    ----------
    predictions : numpy.ndarray
        array containing predictions, last dimension is time
    first_modes_to_remove : int, optional
        Number of low-frequency eigenmodes to remove (highpass)
    last_modes_to_remove_percent : int, optional
        Percentage of high-frequency eigenmodes to remove (lowpass)
    add_mean : bool, optional
        whether to add the mean of the time-courses back to the signal after filtering
        (True, default) or not (False)
    task_lengths : list of ints, optional
        If there are multiple tasks, specify their lengths in TRs. The default is None.
    task_names : list of str, optional
        Task names. The default is None.
    late_iso_dict : dict, optional 
        Dictionary whose keys correspond to task_names. Entries are ndarrays
        containing the TR indices used to compute the BOLD baseline for each task.
        The default is None.

    Returns
    -------
    numpy.ndarray
        filtered version of the array
    """


    if task_lengths is None:
        task_lengths = [predictions.shape[-1]]

    # first assess that the number and sizes of chunks are compatible with the predictions
    assert np.sum(task_lengths) == predictions.shape[-1], "Task lengths \
    are incompatible with the number of prediction timepoints."

    baselines = dict()
    filtered_predictions = np.zeros_like(predictions)

    start = 0
    for i, task_length in enumerate(task_lengths):

        stop = start+task_length

        try:
            coeffs = sp.fft.dct(predictions, norm='ortho', axis=-1)
            coeffs[:, :first_modes_to_remove] = 0
            if last_modes_to_remove_percent>0:
                last_modes_to_remove = int(task_length*last_modes_to_remove_percent/100)
                coeffs[:, -last_modes_to_remove:] = 0
        
            filtered_predictions[..., start:stop] = sp.fft.idct(coeffs, norm='ortho', axis=-1)
        except:
            print("Error occurred during predictions discrete cosine filtering.\
                  Using unfiltered prediction instead")
            filtered_predictions = predictions
        
        if add_mean:
            filtered_predictions[..., start:stop] += np.mean(
                    predictions[..., start:stop], axis=-1)[..., np.newaxis]
            
        
        if late_iso_dict is not None:
            baselines[task_names[i]] = np.median(filtered_predictions[..., start:stop][...,late_iso_dict[task_names[i]]],
                                               axis=-1)

        start += task_length

    if late_iso_dict is not None:
        baseline_full = np.median([baselines[task_name] for task_name in task_names], axis=0)

        start = 0
        for i, task_length in enumerate(task_lengths):
            stop = start+task_length
            baseline_diff = baseline_full - baselines[task_names[i]]
            filtered_predictions[..., start:stop] += baseline_diff[...,np.newaxis]
            start += task_length
        
        filtered_predictions -= baseline_full[...,np.newaxis]

    return filtered_predictions



def sgfilter_predictions(predictions, window_length=201, polyorder=3,
                         highpass=True, add_mean=True, task_lengths=None,
                         task_names=None, late_iso_dict=None, **kwargs):
    """sgfilter_predictions

    savitzky golay filter predictions, to conform to data filtering

    Parameters
    ----------
    predictions : numpy.ndarray
        array containing predictions, last dimension is time
    window_length : int, optional
        window length for SG filter (the default is 201, which is ok for prf experiments, and 
        a bit long for event-related experiments)
    polyorder : int, optional
        polynomial order for SG filter (the default is 3, which performs well for fMRI signals
        when the window length is longer than 2 minutes)
    highpass : bool, optional
        whether to use the sgfilter as highpass (True, default) or lowpass (False)
    add_mean : bool, optional
        whether to add the mean of the time-courses back to the signal after filtering
        (True, default) or not (False)
    task_lengths : list of ints, optional
        If there are multiple tasks, specify their lengths in TRs. The default is None.
    task_names : list of str, optional
        Task names. The default is None.
    late_iso_dict : dict, optional 
        Dictionary whose keys correspond to task_names. Entries are ndarrays
        containing the TR indices used to compute the BOLD baseline for each task.
        The default is None.

    Raises
    ------
    ValueError
        when window_length is even

    Returns
    -------
    numpy.ndarray
        filtered version of the array
    """
    if window_length != 'adaptive':
        if window_length % 2 != 1:
            raise ValueError  # window_length should be odd

    if task_lengths is None:
        task_lengths = [predictions.shape[-1]]

    # first assess that the number and sizes of chunks are compatible with the predictions
    assert np.sum(task_lengths) == predictions.shape[-1], "Task lengths \
    are incompatible with the number of prediction timepoints."

    lp_filtered_predictions = np.zeros_like(predictions)
    if highpass:
        hp_filtered_predictions = np.zeros_like(predictions)

    baselines = dict()

    start = 0
    for i, task_length in enumerate(task_lengths):

        if window_length == 'adaptive':
            if task_length % 2 != 1:
                current_window_length = task_length - 1
            else:
                current_window_length = task_length
        else:
            current_window_length = window_length

        stop = start+task_length

        try:
            lp_filtered_predictions[..., start:stop] = signal.savgol_filter(
            predictions[..., start:stop], window_length=current_window_length,
            polyorder=polyorder)
        except:
            print("Error occurred during predictions savgol filtering.\
                  Using unfiltered prediction instead")

        if add_mean:
            if highpass:
                lp_filtered_predictions[..., start:stop] -= np.mean(
                    predictions[..., start:stop], axis=-1)[..., np.newaxis]
            else:
                lp_filtered_predictions[..., start:stop] += np.mean(
                    predictions[..., start:stop], axis=-1)[..., np.newaxis]

        if highpass:
            hp_filtered_predictions[..., start:stop] = predictions[..., start:stop]\
                - lp_filtered_predictions[..., start:stop]

            if late_iso_dict is not None:
                baselines[task_names[i]] = np.median(hp_filtered_predictions[..., start:stop][...,late_iso_dict[task_names[i]]],
                                                   axis=-1)

        start += task_length

    if late_iso_dict is not None and highpass:
        baseline_full = np.median([baselines[task_name] for task_name in task_names], axis=0)

        start = 0
        for i, task_length in enumerate(task_lengths):
            stop = start+task_length
            baseline_diff = baseline_full - baselines[task_names[i]]
            hp_filtered_predictions[..., start:stop] += baseline_diff[...,np.newaxis]
            start += task_length
        
        hp_filtered_predictions -= baseline_full[...,np.newaxis]

    if highpass:
        return hp_filtered_predictions
    else:
        return lp_filtered_predictions

    
    
def generate_random_legendre_drifts(dimensions=(1000, 120),
                                    amplitude_ranges=[[500, 600], [-50, 50], [-20, 20], [-10, 10], [-5, 5]]):
    """generate_random_legendre_drifts

    generate_random_legendre_drifts generates random slow drifts

    Parameters
    ----------
    dimensions : tuple, optional
        shape of the desired data, latter dimension = timepoints 
        the default is (1000,120), which creates 1000 timecourses for a brief fMRI run
    amplitude_ranges : list, optional
        Amplitudes of each of the components. Ideally, this should follow something like 1/f. 
        the default is [[500,600],[-50,50],[-20,20],[-10,10],[-5,5]]

    Returns
    -------
    numpy.ndarray
        legendre poly drifts with dimensions [dimensions]
    numpy.ndarray
        random multiplication factors that created the drifts
    """
    nr_polys = len(amplitude_ranges)
    drifts = np.polynomial.legendre.legval(
        x=np.arange(dimensions[-1]), c=np.eye(nr_polys)).T
    drifts = (drifts-drifts.mean(0))/drifts.mean(0)
    drifts[:, 0] = np.ones(drifts[:, 0].shape)
    random_factors = np.array([ar[0] + (ar[1]-ar[0])/2.0 + (np.random.rand(dimensions[0])-0.5) * (ar[1]-ar[0])
                               for ar in amplitude_ranges])
    return np.dot(drifts, random_factors), random_factors


def generate_random_cosine_drifts(dimensions=(1000, 120),
                                  amplitude_ranges=[[500, 600], [-50, 50], [-20, 20], [-10, 10], [-5, 5]]):
    """generate_random_cosine_drifts

    generate_random_cosine_drifts generates random slow drifts

    Parameters
    ----------
    dimensions : tuple, optional
        shape of the desired data, latter dimension = timepoints 
        the default is (1000,120), which creates 1000 timecourses for a brief fMRI run
    amplitude_ranges : list, optional
        Amplitudes of each of the components. Ideally, this should follow something like 1/f. 
        the default is [[500,600],[-50,50],[-20,20],[-10,10],[-5,5]]

    Returns
    -------
    numpy.ndarray
        discrete cosine drifts with dimensions [dimensions]
    numpy.ndarray
        random multiplication factors that created the drifts
    """
    nr_freqs = len(amplitude_ranges)
    x = np.linspace(0, np.pi, dimensions[-1])
    drifts = np.array([np.cos(x*f) for f in range(nr_freqs)]).T
    random_factors = np.array([ar[0] + (ar[1]-ar[0])/2.0 + (np.random.rand(dimensions[0])-0.5) * (ar[1]-ar[0])
                               for ar in amplitude_ranges])
    return np.dot(drifts, random_factors), random_factors


def generate_arima_noise(ar=(1, 0.4),
                         ma=(1, 0.0),
                         dimensions=(1000, 120),
                         **kwargs):
    """generate_arima_noise

    generate_arima_noise creates temporally correlated noise

    Parameters
    ----------
    ar : tuple, optional
        arima autoregression parameters for statsmodels generation of noise 
        (the default is (1,0.4), which should be a reasonable setting for fMRI noise)
    ma : tuple, optional
        arima moving average parameters for statsmodels generation of noise 
        (the default is (1,0.0), which should be a reasonable setting for fMRI noise)        
    dimensions : tuple, optional
        the first dimension is the nr of separate timecourses, the second dimension
        is the timeseries length.
        (the default is (1000,120), a reasonable if brief length for an fMRI run)

    **kwargs are passed on to statsmodels.tsa.arima_process.arma_generate_sample

    Returns 
    -------
    numpy.ndarray
        noise of requested dimensions and properties

    """
    return np.array([arma_generate_sample(ar, ma, dimensions[1], **kwargs) for _ in range(dimensions[0])])
