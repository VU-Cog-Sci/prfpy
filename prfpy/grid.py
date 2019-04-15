import numpy as np
from .rf import gauss2D_iso_cart   # import required RF shapes
from .timecourse import stimulus_through_prf, \
    convolve_stimulus_dm, \
    generate_random_cosine_drifts, \
    generate_arima_noise, \
    sgfilter_predictions
from .stimulus import PRFStimulus2D
from hrf_estimation.hrf import spmt, dspmt, ddspmt


class Gridder(object):
    """Gridder

    Class that takes care of generating grids for pRF fitting and simulations
    """

    def __init__(self, stimulus):
        """__init__

        constructor for Gridder, takes stimulus object as argument

        Parameters
        ----------
        stimulus : PRFStimulus2D or PRFStimulusDD
            Stimulus object containing information about the stimulus, 
            and the space in which it lives.

        """
        self.stimulus = stimulus


class Iso2DGaussianGridder(Gridder):
    """Iso2DGaussianGridder
    To extend please create a setup_XXX_grid function for any new way of 
    defining grids.
    """

    def __init__(self,
                 stimulus,
                 hrf=None,
                 filter_predictions=False,
                 window_length=201,
                 polyorder=3,
                 highpass=True,
                 **kwargs):
        """__init__ for Iso2DGaussianGridder

        constructor, sets up stimulus and hrf for this gridder

        Parameters
        ----------
        stimulus : PRFStimulus2D
            Stimulus object specifying the information about the stimulus, 
            and the space in which it lives.
        hrf : string, list or numpy.ndarray, optional
            HRF shape for this gridder. 
            Can be 'direct', which implements nothing (for eCoG or later convolution),
            a list or array of 3, which are multiplied with the three spm HRF basis functions,
            and an array already sampled on the TR by the user.
            (the default is None, which implements standard spm HRF)
        filter_predictions : boolean, optional 
            whether to high-pass filter the predictions, default False
        window_length : int, odd number, optional 
            length of savgol filter, default 201 TRs 
        polyorder : int, optional  
            polynomial order of savgol filter, default 3
        highpass : boolean, optional
            whether to filter highpass or lowpass, default True
        """
        super().__init__(stimulus)
        self.__dict__.update(kwargs)

        # HRF stuff
        if hrf == None:  # for use with standard fMRI
            hrf_times = np.linspace(0, 40, 40/self.stimulus.TR, endpoint=False)
            self.hrf = spmt(hrf_times)
        elif hrf == 'direct':  # for use with anything like eCoG with instantaneous irf
            self.hrf = np.array([1])
        # some specific hrf with spm basis set
        elif ((type(hrf) == list) or (type(hrf) == np.ndarray)) and len(hrf) == 3:
            hrf_times = np.linspace(0, 40, 40/self.stimulus.TR, endpoint=False)
            self.hrf = np.array([hrf[0] * spmt(hrf_times),
                                 hrf[1] * dspmt(hrf_times),
                                 hrf[2] * ddspmt(hrf_times)]).sum(axis=0)
        # some specific hrf already defined at the TR (!)
        elif type(hrf) == np.ndarray and len(hrf) > 3:
            self.hrf = hrf

        self.convolved_design_matrix = convolve_stimulus_dm(
            stimulus.design_matrix, hrf=self.hrf)

        # filtering stuff
        self.filter_predictions = filter_predictions
        self.window_length = window_length
        self.polyorder = polyorder
        self.highpass = highpass

    def create_rfs(self):
        """create_rfs

        creates rfs for the grid

        """
        assert hasattr(self, 'xs'), "please set up the grid first"
        self.grid_rfs = gauss2D_iso_cart(
            x=self.stimulus.x_coordinates[..., np.newaxis],
            y=self.stimulus.y_coordinates[..., np.newaxis],
            mu=np.array([self.xs, self.ys]).reshape((-1, 2)).T,
            sigma=self.sizes.ravel())
        # won't have to perform exponentiation if all ns are one (the default value)
        if len(np.unique(self.ns)) != 1:
            self.grid_rfs **= self.ns.ravel()
        self.grid_rfs = self.grid_rfs.T

    def stimulus_times_prfs(self):
        """stimulus_times_prfs

        creates timecourses for each of the rfs in self.grid_rfs

        """
        assert hasattr(self, 'grid_rfs'), "please create the rfs first"
        self.predictions = stimulus_through_prf(
            self.grid_rfs, self.convolved_design_matrix)

        # normalize the resulting predictions to peak value of 1
        # self.predictions /= self.predictions.max(axis=-1)[:, np.newaxis]

    def create_grid_predictions(self,
                                ecc_grid,
                                polar_grid,
                                size_grid,
                                n_grid=[1]):
        """create_predictions

        creates predictions for a given set of parameters

        [description]

        Parameters
        ----------
        ecc_grid : list
            to be filled in by user
        polar_grid : list
            to be filled in by user
        size_grid : list
            to be filled in by user
        n_grid : list, optional
            to be filled in by user 
            (the default is [1])
        """
        assert ecc_grid is not None and polar_grid is not None and size_grid is not None, \
            "please fill in all spatial grids"

        self.eccs, self.polars, self.sizes, self.ns = np.meshgrid(
            ecc_grid, polar_grid, size_grid, n_grid)
        self.xs, self.ys = np.sin(self.polars) * \
            self.eccs, np.cos(self.polars) * self.eccs

        self.create_rfs()
        self.stimulus_times_prfs()

        if self.filter_predictions:
            self.predictions = sgfilter_predictions(self.predictions.T,
                                                    window_length=window_length,
                                                    polyorder=polyorder,
                                                    highpass=highpass,
                                                    **kwargs).T
            self.filtered_predictions = True
        else:
            self.filtered_predictions = False

    def return_single_prediction(self,
                                 mu_x,
                                 mu_y,
                                 size,
                                 beta=1.0,
                                 baseline=0.0,
                                 n=1.0):
        """return_single_prediction

        returns the prediction for a single set of parameters.
        As this is to be used during iterative search, it also 
        has arguments beta and baseline. 

        Parameters
        ----------
        mu_x : float
            x-position of pRF
        mu_y : float
            y-position of pRF
        size : float
            size of pRF
        beta : float, optional
            amplitude of pRF (the default is 1)        
        baseline : float, optional
            baseline of pRF (the default is 0)
        n : float, optional
            exponent of pRF (the default is 1, which is a linear Gaussian)        

        Returns
        -------
        numpy.ndarray
            single prediction given the model
        """
        # create the single rf
        rf = gauss2D_iso_cart(x=self.stimulus.x_coordinates[..., np.newaxis],
                              y=self.stimulus.y_coordinates[..., np.newaxis],
                              mu=(mu_x, mu_y),
                              sigma=size)
        # won't have to perform exponentiation if n == 1
        if n != 1:
            rf **= n
        rf = rf.T
        # create timecourse
        tc = stimulus_through_prf(rf, self.convolved_design_matrix)
        # tc /= tc.max()
        if not self.filter_predictions:
            return baseline + beta * tc
        else:
            return baseline + beta * sgfilter_predictions(tc.T,
                                                          window_length=window_length,
                                                          polyorder=polyorder,
                                                          highpass=highpass,
                                                          **kwargs).T

    def create_drifts_and_noise(self,
                                drift_ranges=[[0, 0]],
                                noise_ar=None,
                                noise_amplitude=1.0):
        """add_drifs_and_noise

        creates noise and drifts of size equal to the predictions

        Parameters
        ----------
        drift_ranges : list of 2-lists of floats, optional
            specifies the lower- and upper bounds of the  ranges 
            of each of the discrete cosine low-pass components
            to be generated
        noise_ar : 2x2 list.
            argument passed to timecourse.generate_arima_noise
            (the default is None, for no noise)
        noise_amplitude : float, optional

        """
        assert hasattr(
            self, 'predictions'), "please first create the grid to which to add noise"
        self.random_drifts = generate_random_cosine_drifts(
            dimensions=self.predictions.shape, amplitude_ranges=drift_ranges)
        if noise_ar is not None:
            self.random_noise = generate_arima_noise(
                ar=noise_ar[0], ma=noise_ar[1], dimensions=self.predictions.shape) * noise_amplitude
        else:
            self.random_noise = np.zeros_like(self.predictions)
