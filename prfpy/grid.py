import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal
from nistats.hemodynamic_models import spm_hrf, spm_time_derivative, spm_dispersion_derivative

from .rf import gauss2D_iso_cart   # import required RF shapes
from .timecourse import stimulus_through_prf, \
    convolve_stimulus_dm, \
    generate_random_cosine_drifts, \
    generate_arima_noise, \
    sgfilter_predictions


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

    def create_hrf(self, hrf_params=[1.0, 1.0, 0.0]):
        """
        
        construct single or multiple HRFs        

        Parameters
        ----------
        hrf_params : TYPE, optional
            DESCRIPTION. The default is [1.0, 1.0, 0.0].

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        hrf = np.array(
            [
                np.ones_like(hrf_params[1])*hrf_params[0] *
                spm_hrf(
                    tr=self.stimulus.TR,
                    oversampling=1,
                    time_length=40)[...,np.newaxis],
                hrf_params[1] *
                spm_time_derivative(
                    tr=self.stimulus.TR,
                    oversampling=1,
                    time_length=40)[...,np.newaxis],
                hrf_params[2] *
                spm_dispersion_derivative(
                    tr=self.stimulus.TR,
                    oversampling=1,
                    time_length=40)[...,np.newaxis]]).sum(
            axis=0)                    

        return hrf.T
    
    def convolve_timecourse_hrf(self, tc, hrf):
        """
        
        Convolve neural timecourses with single or multiple hrfs.

        Parameters
        ----------
        tc : TYPE
            DESCRIPTION.
        hrf : TYPE
            DESCRIPTION.

        Returns
        -------
        convolved_tc : TYPE
            DESCRIPTION.

        """
        #scipy fftconvolve does not have padding options so doing it manually
        pad_length = 20
        pad = np.tile(tc[:,0], (pad_length,1)).T
        padded_tc = np.hstack((pad,tc))
        
        multi_hrf = True
        
        # use median HRF when multiple are provided
        if hrf.shape[0]>1:           
            assert hrf.shape[0] == tc.shape[0], f"{hrf.shape[0]} HRFs provided vs {tc.shape[0]} timecourses"
            
            if not multi_hrf:             
                hrf = np.median(hrf, axis=0).reshape(1,-1)
                convolved_tc = signal.fftconvolve(padded_tc, hrf, axes=(-1))[..., pad_length:tc.shape[-1]+pad_length]
                
            else:                 
                convolved_tc = np.zeros_like(tc)
                
                for n_ in range(hrf.shape[0]):
                    convolved_tc[n_,:] = signal.fftconvolve(padded_tc[n_,:],hrf[n_,:])[..., pad_length:tc.shape[-1]+pad_length] 
                    
        else:
            convolved_tc = signal.fftconvolve(padded_tc, hrf, axes=(-1))[..., pad_length:tc.shape[-1]+pad_length] 

        return convolved_tc

    def create_drifts_and_noise(self,
                                drift_ranges=[[0, 0]],
                                noise_ar=None,
                                noise_ma=(1, 0.0),
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
                ar=noise_ar, ma=noise_ma, dimensions=self.predictions.shape) * noise_amplitude
        else:
            self.random_noise = np.zeros_like(self.predictions)


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
                 add_mean=True,
                 normalize_RFs=False,
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
        add_mean : boolean, optional
            whether to add mean to filtered predictions, default True
        task_lengths : list or numpy.ndarray, optional
            specify length of each condition in TRs
            If not None, the predictions are split in the time dimension in len(task_lengths) chunks,
            and the savgol filter is applied to each chunk separately.
            The i^th chunk has size task_lengths[i]
        """
        super().__init__(stimulus)
        self.__dict__.update(kwargs)

        # HRF stuff
        if hrf is None:  # for use with standard fMRI
            self.hrf = spm_hrf(tr=self.stimulus.TR,
                               oversampling=1, time_length=40)
        elif hrf == 'direct':  # for use with anything like eCoG with instantaneous irf
            self.hrf = np.array([1])
        # some specific hrf with spm basis set
        elif ((isinstance(hrf, list)) or (isinstance(hrf, np.ndarray))) and len(hrf) == 3:
            self.hrf = self.create_hrf(hrf_params=hrf)
        # some specific hrf already defined at the TR (!)
        elif isinstance(hrf, np.ndarray) and len(hrf) > 3:
            self.hrf = hrf

        self.stimulus.convolved_design_matrix = convolve_stimulus_dm(
            stimulus.design_matrix, hrf=self.hrf)

        # filtering and other stuff
        self.filter_predictions = filter_predictions
        self.window_length = window_length
        self.polyorder = polyorder
        self.highpass = highpass
        self.add_mean = add_mean
        
        #normalizing RFs to have volume 1
        self.normalize_RFs = normalize_RFs
        

        

    def create_rfs(self):
        """create_rfs

        creates rfs for the grid

        """
        assert hasattr(self, 'xs'), "please set up the grid first"
        self.grid_rfs = np.rot90(gauss2D_iso_cart(
            x=self.stimulus.x_coordinates[..., np.newaxis],
            y=self.stimulus.y_coordinates[..., np.newaxis],
            mu=np.array([self.xs.ravel(), self.ys.ravel()]),
            sigma=self.sizes.ravel(),
            normalize_RFs=self.normalize_RFs).T, axes=(1,2))

    def stimulus_times_prfs(self):
        """stimulus_times_prfs

        creates timecourses for each of the rfs in self.grid_rfs

        """
        assert hasattr(self, 'grid_rfs'), "please create the rfs first"
        self.predictions = stimulus_through_prf(
            self.grid_rfs, self.stimulus.convolved_design_matrix)


    def create_grid_predictions(self,
                                ecc_grid,
                                polar_grid,
                                size_grid):
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
        """
        assert ecc_grid is not None and polar_grid is not None and size_grid is not None, \
            "please fill in all spatial grids"

        self.eccs, self.polars, self.sizes = np.meshgrid(
            ecc_grid, polar_grid, size_grid)
        self.xs, self.ys = np.cos(self.polars) * \
            self.eccs, np.sin(self.polars) * self.eccs

        self.create_rfs()
        self.stimulus_times_prfs()

        if self.filter_predictions:
            self.predictions = sgfilter_predictions(
                self.predictions,
                window_length=self.window_length,
                polyorder=self.polyorder,
                highpass=self.highpass,
                add_mean=self.add_mean,
                task_lengths=self.stimulus.task_lengths, 
                task_names=self.stimulus.task_names, 
                late_iso_dict=self.stimulus.late_iso_dict)
            self.filtered_predictions = True
        else:
            self.filtered_predictions = False

    def return_prediction(self,
                                 mu_x,
                                 mu_y,
                                 size,
                                 beta,
                                 baseline,
                                 hrf_1=None,
                                 hrf_2=None):
        """return_prediction

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
        beta : float
            amplitude of pRF
        baseline : float
            baseline of pRF

        Returns
        -------
        numpy.ndarray
            single prediction given the model
        """
        if hrf_1 is None or hrf_2 is None:
            current_hrf = self.hrf
        else:
            current_hrf = self.create_hrf([1.0, hrf_1, hrf_2])

        # create the single rf
        rf = np.rot90(gauss2D_iso_cart(x=self.stimulus.x_coordinates[..., np.newaxis],
                              y=self.stimulus.y_coordinates[..., np.newaxis],
                              mu=(mu_x, mu_y),
                              sigma=size,
                              normalize_RFs=self.normalize_RFs).T, axes=(1,2))

        dm = self.stimulus.design_matrix
        neural_tc = stimulus_through_prf(rf, dm)


        tc = self.convolve_timecourse_hrf(neural_tc, current_hrf)
        

        if not self.filter_predictions:
            return baseline[..., np.newaxis] + beta[..., np.newaxis] * tc
        else:
            return baseline[..., np.newaxis] + beta[..., np.newaxis] * sgfilter_predictions(
                tc,
                window_length=self.window_length,
                polyorder=self.polyorder,
                highpass=self.highpass,
                add_mean=self.add_mean,
                task_lengths=self.stimulus.task_lengths,
                task_names=self.stimulus.task_names,
                late_iso_dict=self.stimulus.late_iso_dict)


class CSS_Iso2DGaussianGridder(Iso2DGaussianGridder):

    def return_prediction(self,
                                 mu_x,
                                 mu_y,
                                 size,
                                 beta,
                                 baseline,
                                 n,
                                 hrf_1=None,
                                 hrf_2=None):
        """return_prediction

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

        if hrf_1 is None or hrf_2 is None:
            current_hrf = self.hrf
        else:
            current_hrf = self.create_hrf([1.0, hrf_1, hrf_2])

        # create the single rf
        rf = np.rot90(gauss2D_iso_cart(x=self.stimulus.x_coordinates[..., np.newaxis],
                              y=self.stimulus.y_coordinates[..., np.newaxis],
                              mu=(mu_x, mu_y),
                              sigma=size,
                              normalize_RFs=self.normalize_RFs).T, axes=(1,2))

        dm = self.stimulus.design_matrix
        neural_tc = stimulus_through_prf(rf, dm)**n[..., np.newaxis]
        
        tc = self.convolve_timecourse_hrf(neural_tc, current_hrf)

        if not self.filter_predictions:
            return baseline[..., np.newaxis] + beta[..., np.newaxis] * tc
        else:
            return baseline[..., np.newaxis] + beta[..., np.newaxis] * sgfilter_predictions(
                tc,
                window_length=self.window_length,
                polyorder=self.polyorder,
                highpass=self.highpass,
                add_mean=self.add_mean,
                task_lengths=self.stimulus.task_lengths, 
                task_names=self.stimulus.task_names, 
                late_iso_dict=self.stimulus.late_iso_dict)


class Norm_Iso2DGaussianGridder(Iso2DGaussianGridder):
    """Norm_Iso2DGaussianGridder

    Redefining class for normalization model

    """

    def create_grid_predictions(self,
                                gaussian_params,
                                n_predictions,
                                n_timepoints,
                                sa,
                                ss,
                                nb,
                                sb):
        """create_predictions

        creates predictions for a given set of parameters

        [description]

        Parameters
        ----------
        gaussian_params: array size (3), containing prf position and size.
        n_predictions, n_timepoints: self explanatory, obtained from fitter
        nb,sa,ss,sb: meshgrid, created in fitter.grid_fit

        """

        predictions = np.zeros((n_predictions, n_timepoints), dtype='float32')    
        
        for idx in range(n_predictions):
            prediction_params = np.array([gaussian_params[0],
                                    gaussian_params[1],
                                    gaussian_params[2],
                                    1.0,
                                    0.0,
                                    sa[idx],
                                    ss[idx],
                                    nb[idx],
                                    sb[idx]]).T
            predictions[idx,
                        :] = self.return_prediction(*list(prediction_params)).astype('float32')

        return predictions

    def return_prediction(self,
                                 mu_x,
                                 mu_y,
                                 prf_size,
                                 prf_amplitude,
                                 bold_baseline,
                                 srf_amplitude,
                                 srf_size,
                                 neural_baseline,
                                 surround_baseline,
                                 hrf_1=None,
                                 hrf_2=None
                                 ):
        """return_prediction [summary]

        returns the prediction for a single set of parameters.

        Parameters
        ----------
        mu_x : [type]
            [description]
        mu_y : [type]
            [description]
        prf_size : [type]
            [description]
        prf_amplitude : [type]
            [description]
        bold_baseline : [type]
            [description]
        neural_baseline : [type]
            [description]
        srf_amplitude : [type]
            [description]
        srf_size : [type]
            [description]
        surround_baseline : [type]
            [description]


        Returns
        -------
        numpy.ndarray
            single prediction given the model
        """

        if hrf_1 is None or hrf_2 is None:
            current_hrf = self.hrf
        else:
            current_hrf = self.create_hrf([1.0, hrf_1, hrf_2])

        # create the rfs

        prf = np.rot90(gauss2D_iso_cart(x=self.stimulus.x_coordinates[..., np.newaxis],
                               y=self.stimulus.y_coordinates[..., np.newaxis],
                               mu=(mu_x, mu_y),
                               sigma=prf_size,
                               normalize_RFs=self.normalize_RFs).T, axes=(1,2))

        # surround receptive field (denominator)
        srf = np.rot90(gauss2D_iso_cart(x=self.stimulus.x_coordinates[..., np.newaxis],
                               y=self.stimulus.y_coordinates[..., np.newaxis],
                               mu=(mu_x, mu_y),
                               sigma=srf_size,
                               normalize_RFs=self.normalize_RFs).T, axes=(1,2))

        dm = self.stimulus.design_matrix

        # create normalization model timecourse
        neural_tc = (prf_amplitude[..., np.newaxis] * stimulus_through_prf(prf, dm) + neural_baseline[..., np.newaxis]) /\
            (srf_amplitude[..., np.newaxis] * stimulus_through_prf(srf, dm) + surround_baseline[..., np.newaxis])

        tc = self.convolve_timecourse_hrf(neural_tc, current_hrf)
                
        if not self.filter_predictions:
            return bold_baseline[..., np.newaxis] + tc
        else:
            return bold_baseline[..., np.newaxis] + sgfilter_predictions(
                tc,
                window_length=self.window_length,
                polyorder=self.polyorder,
                highpass=self.highpass,
                add_mean=self.add_mean,
                task_lengths=self.stimulus.task_lengths, 
                task_names=self.stimulus.task_names, 
                late_iso_dict=self.stimulus.late_iso_dict)



class DoG_Iso2DGaussianGridder(Iso2DGaussianGridder):
    """redefining class for difference of Gaussians in iterative fit.
    """

    def return_prediction(self,
                                 mu_x,
                                 mu_y,
                                 prf_size,
                                 prf_amplitude,
                                 bold_baseline,

                                 srf_amplitude,
                                 srf_size,
                                 hrf_1=None,
                                 hrf_2=None
                                 ):
        """return_prediction

        returns the prediction for a single set of parameters.
        As this is to be used during iterative search, it also
        has arguments beta and baseline.

        Parameters
        ----------
        mu_x : float
            x-position of pRF
        mu_y : float
            y-position of pRF
        prf_size : float
            size of pRF


        Returns
        -------
        numpy.ndarray
            single prediction given the model
        """
        if hrf_1 is None or hrf_2 is None:
            current_hrf = self.hrf
        else:
            current_hrf = self.create_hrf([1.0, hrf_1, hrf_2])
        # create the rfs
        prf = np.rot90(gauss2D_iso_cart(x=self.stimulus.x_coordinates[..., np.newaxis],
                               y=self.stimulus.y_coordinates[..., np.newaxis],
                               mu=(mu_x, mu_y),
                               sigma=prf_size,
                              normalize_RFs=self.normalize_RFs).T, axes=(1,2))

        # surround receptive field
        srf = np.rot90(gauss2D_iso_cart(x=self.stimulus.x_coordinates[..., np.newaxis],
                               y=self.stimulus.y_coordinates[..., np.newaxis],
                               mu=(mu_x, mu_y),
                               sigma=srf_size,
                              normalize_RFs=self.normalize_RFs).T, axes=(1,2))

        dm = self.stimulus.design_matrix

        neural_tc = prf_amplitude[..., np.newaxis] * stimulus_through_prf(prf, dm) - \
            srf_amplitude[..., np.newaxis] * stimulus_through_prf(srf, dm)

        tc = self.convolve_timecourse_hrf(neural_tc, current_hrf)

        if not self.filter_predictions:
            return bold_baseline[..., np.newaxis] + tc
        else:
            return bold_baseline[..., np.newaxis] + sgfilter_predictions(
                tc,
                window_length=self.window_length,
                polyorder=self.polyorder,
                highpass=self.highpass,
                add_mean=self.add_mean,
                task_lengths=self.stimulus.task_lengths, 
                task_names=self.stimulus.task_names, 
                late_iso_dict=self.stimulus.late_iso_dict)
