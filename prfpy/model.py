import warnings
import numpy as np
import scipy.signal as signal
import warnings
from nilearn.glm.first_level.hemodynamic_models import spm_hrf, spm_time_derivative, spm_dispersion_derivative

from .rf import gauss2D_iso_cart, gauss1D_cart  # import required RF shapes
from .timecourse import stimulus_through_prf, \
    convolve_stimulus_dm, \
    generate_random_cosine_drifts, \
    generate_arima_noise, \
    filter_predictions


class Model(object):
    """Model

    Class that takes care of generating grids for pRF fitting and simulations
    """

    def __init__(self, stimulus):
        """__init__

        constructor for Model, takes stimulus object as argument

        Parameters
        ----------
        stimulus : PRFStimulus2D or PRFStimulusDD
            Stimulus object containing information about the stimulus,
            and the space in which it lives.

        """
        self.stimulus = stimulus

    def convolve_timecourse_hrf(self, tc, hrf_values: np.ndarray):
        """

        Convolve neural timecourses with single or multiple hrfs.

        Parameters
        ----------
        tc : ndarray, 1D or 2D
            The timecourse(s) to be convolved.
        hrf : ndarray, 1D or 2D
            The HRF. Can be single, or a different one for each timecourse.

        Returns
        -------
        convolved_tc : ndarray
            Convolved timecourse.

        """
        # scipy fftconvolve does not have padding options so doing it manually
        pad_length = 20
        pad = np.tile(tc[:, 0], (pad_length, 1)).T
        padded_tc = np.hstack((pad, tc))

        if hrf_values.shape[0] > 1:
            assert hrf_values.shape[0] == tc.shape[
                0], f"{hrf_values.shape[0]} HRFs provided vs {tc.shape[0]} timecourses"
            median_hrf = np.median(hrf_values, axis=0).reshape(1, -1)
            if np.all([np.allclose(median_hrf, single_hrf_values.reshape(1, -1)) for single_hrf_values in hrf_values]):

                convolved_tc = signal.fftconvolve(
                    padded_tc, median_hrf, axes=(-1))[..., pad_length:tc.shape[-1]+pad_length]

            else:
                convolved_tc = np.zeros_like(tc)

                for n_ in range(hrf_values.shape[0]):
                    convolved_tc[n_, :] = signal.fftconvolve(
                        padded_tc[n_, :], hrf_values[n_, :])[..., pad_length:tc.shape[-1]+pad_length]

        else:
            convolved_tc = signal.fftconvolve(
                padded_tc, hrf_values, axes=(-1))[..., pad_length:tc.shape[-1]+pad_length]

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


class HRF():
    # We want a class that can be used analogous to the hierarchy already present
    # for the model, fitter, stimulus and so forth,
    # thus we want a method creating a standard HRF,
    # one creating a the spm basis set hrf and of course one where values can be user-defined
    # Old documentation:
    # Can be 'direct', which implements nothing (for eCoG or later convolution),
    #         a list or array of 3, which are multiplied with the three spm HRF basis functions,
    #         and an array already sampled on the TR by the user.
    #         (the default is None, which implements standard spm HRF)

    def __init__(self, values: np.ndarray = None):
        if values is not None:
            assert values.ndim > 2, "Number of dimension for HRF values needs to be at least two. Last dimension corresponds to time."
            self.values = values
        else:
            # TODO Should the default here be that create_spm_hrf is called with default params?
            self.values = None


    def _assert_values_filled(self, force = False):
        if not force:
            assert self.hasValues(), "HRF values have already been assigned! Try with force."

    def hasValues(self):
        return self.values is not None and len(self.values.flatten()) > 0
     
    def create_spm_hrf(self, TR, force=False, hrf_params=[1.0, 1.0, 0.0]):
        """construct single or multiple HRFs        

            Parameters
            ----------
            hrf_params : TYPE, optional
                DESCRIPTION. The default is [1.0, 1.0, 0.0].

            Returns
            -------
            hrf : ndarray
                the hrf.
        """
        self._assert_values_filled(force)

        assert len(hrf_params) == 3

        values = np.array(
            [
                np.ones_like(hrf_params[1])*hrf_params[0] *
                spm_hrf(
                    tr=TR,
                    oversampling=1,
                    time_length=40)[..., np.newaxis],
                hrf_params[1] *
                spm_time_derivative(
                    tr=TR,
                    oversampling=1,
                    time_length=40)[..., np.newaxis],
                hrf_params[2] *
                spm_dispersion_derivative(
                    tr=TR,
                    oversampling=1,
                    time_length=40)[..., np.newaxis]]).sum(
            axis=0)

        self.values = values.T

    def create_direct_hrf(self, force=False):
        # Is this method necessary or can this case be assumed to be input be the user through the constructor?
        self._assert_values_filled(force)

        self.values = np.array([[1]])

class Iso2DGaussianModel(Model):
    """Iso2DGaussianModel
    To extend please create a setup_XXX_grid function for any new way of
    defining grids.
    """

    def __init__(self,
                 stimulus,
                 hrf: HRF = None,
                 filter_predictions=False,
                 filter_type='dc',
                 filter_params={},
                 normalize_RFs=False,
                 **kwargs):
        """__init__ for Iso2DGaussianModel

        constructor, sets up stimulus and hrf for this Model

        Parameters
        ----------
        stimulus : PRFStimulus2D
            Stimulus object specifying the information about the stimulus,
            and the space in which it lives.
        hrf : HRF, array-like, None
            HRF for this Model.
        filter_predictions : boolean, optional
            whether to high-pass filter the predictions, default False
        filter_type, filter_params : see timecourse.py
        normalize_RFs : whether or not to normalize the RF volumes (generally not needed).
        """
        super().__init__(stimulus)
        self.__dict__.update(kwargs)


        # make HRF class downwards compatible
        self.hrf = HRF()

        if hrf is None:
            self.hrf.create_direct_hrf(force=True)
            print("Using no HRF")

        elif (type(hrf) is np.ndarray or type(hrf) is list) and len(hrf) == 3:
            self.hrf.create_spm_hrf(hrf_params=hrf, force=True, TR=self.tr)
            warnings.warn("Specifying HRF parameters is deprecated. Please refer to the HRF class and specify an HRF object.", FutureWarning)

        elif type(hrf) is np.ndarray and hrf.shape[0] == 1 and hrf.shape[1] > 3:
            self.hrf = HRF(values=hrf)
            warnings.warn("Specifying HRF values is deprecated. Please refer to the HRF class and specify an HRF object.", FutureWarning)

        elif type(hrf) is HRF:
            # this should be the only way used in the future implying that the user specifies that HRF object beforehand!
            self.hrf = hrf

        assert self.hrf.hasValues(), "Initialize HRF values first!"

        self.stimulus.convolved_design_matrix = convolve_stimulus_dm(
            stimulus.design_matrix, hrf_values=self.hrf.values)

        # filtering and other stuff
        self.filter_predictions = filter_predictions
        self.filter_type = filter_type

        # settings for filter
        self.filter_params = filter_params

        # adding stimulus parameters
        self.filter_params['task_lengths'] = self.stimulus.task_lengths
        self.filter_params['task_names'] = self.stimulus.task_names
        self.filter_params['late_iso_dict'] = self.stimulus.late_iso_dict

        # normalizing RFs to have volume 1
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
            normalize_RFs=self.normalize_RFs).T, axes=(1, 2))

    def stimulus_times_prfs(self):
        """stimulus_times_prfs

        creates timecourses for each of the rfs in self.grid_rfs

        """
        assert hasattr(self, 'grid_rfs'), "please create the rfs first"
        self.predictions = stimulus_through_prf(
            self.grid_rfs, self.stimulus.convolved_design_matrix,
            self.stimulus.dx)

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
            self.predictions = filter_predictions(
                self.predictions,
                self.filter_type,
                self.filter_params)
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
        hrf_1, hrf_2 : floats, optional
            hrf parameters, specified only if hrf is being fit to data, otherwise not needed.

        Returns
        -------
        numpy.ndarray
            single prediction given the model
        """
        if hrf_1 is not None and hrf_2 is not None:
            current_hrf = self.hrf.create_spm_hrf(
                force=True, TR=self.stimulus.TR, hrf_params=[1.0, hrf_1, hrf_2])
        else:
            assert self.hrf.hasValues(), "Initialize HRF values first!"
            current_hrf = self.hrf.values

        # create the single rf
        rf = np.rot90(gauss2D_iso_cart(x=self.stimulus.x_coordinates[..., np.newaxis],
                                       y=self.stimulus.y_coordinates[...,
                                                                     np.newaxis],
                                       mu=(mu_x, mu_y),
                                       sigma=size,
                                       normalize_RFs=self.normalize_RFs).T, axes=(1, 2))

        dm = self.stimulus.design_matrix
        neural_tc = stimulus_through_prf(rf, dm, self.stimulus.dx)

        tc = self.convolve_timecourse_hrf(neural_tc, current_hrf)

        if not self.filter_predictions:
            return baseline[..., np.newaxis] + beta[..., np.newaxis] * tc
        else:
            return baseline[..., np.newaxis] + beta[..., np.newaxis] * filter_predictions(
                tc,
                self.filter_type,
                self.filter_params)


class CSS_Iso2DGaussianModel(Iso2DGaussianModel):

    def create_grid_predictions(self,
                                gaussian_params,
                                nn):
        """create_predictions

        creates predictions for a given set of parameters

        [description]

        Parameters
        ----------
        gaussian_params: ndarray size (3)
            containing prf position and size.
        nn: ndarrays
            containing the range of grid values for other CSS model parameters
            (exponent)


        """
        n_predictions = len(nn)

        prediction_params = np.array([gaussian_params[0]*np.ones(n_predictions),
                                      gaussian_params[1] *
                                      np.ones(n_predictions),
                                      gaussian_params[2] *
                                      np.ones(n_predictions) * 
                                      np.sqrt(nn),
                                      1.0*np.ones(n_predictions),
                                      0.0*np.ones(n_predictions),
                                      nn])

        return self.return_prediction(*list(prediction_params)).astype('float32')

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
        hrf_1, hrf_2 : floats, optional
            hrf parameters, specified only if hrf is being fit to data, otherwise not needed.

        Returns
        -------
        numpy.ndarray
            single prediction given the model
        """

        if hrf_1 is not None and hrf_2 is not None:
            current_hrf = self.hrf.create_spm_hrf(
                force=True, TR=self.stimulus.TR, hrf_params=[1.0, hrf_1, hrf_2])

        assert self.hrf.hasValues(), "Initialize HRF values first!"
        current_hrf = self.hrf.values

        # create the single rf
        rf = np.rot90(gauss2D_iso_cart(x=self.stimulus.x_coordinates[..., np.newaxis],
                                       y=self.stimulus.y_coordinates[...,
                                                                     np.newaxis],
                                       mu=(mu_x, mu_y),
                                       sigma=size,
                                       normalize_RFs=self.normalize_RFs).T, axes=(1, 2))

        dm = self.stimulus.design_matrix
        neural_tc = stimulus_through_prf(
            rf, dm, self.stimulus.dx)**n[..., np.newaxis]

        tc = self.convolve_timecourse_hrf(neural_tc, current_hrf)

        if not self.filter_predictions:
            return baseline[..., np.newaxis] + beta[..., np.newaxis] * tc
        else:
            return baseline[..., np.newaxis] + beta[..., np.newaxis] * filter_predictions(
                tc,
                self.filter_type,
                self.filter_params)


class Norm_Iso2DGaussianModel(Iso2DGaussianModel):
    """Norm_Iso2DGaussianModel

    Redefining class for normalization model

    """

    def create_grid_predictions(self,
                                gaussian_params,
                                sa,
                                ss,
                                nb,
                                sb):
        """create_predictions

        creates predictions for a given set of parameters

        [description]

        Parameters
        ----------
        gaussian_params: ndarray size (3)
            containing prf position and size.
        sa,ss,nb,sb: ndarrays
            containing the range of grid values for other norm model parameters
            (surroud amplitude (C), surround size (sigma_2), neural baseline (B), surround baseline (D))


        """
        n_predictions = len(sa)

        prediction_params = np.array([gaussian_params[0]*np.ones(n_predictions),
                                      gaussian_params[1] *
                                      np.ones(n_predictions),
                                      gaussian_params[2] *
                                      np.ones(n_predictions),
                                      1.0*np.ones(n_predictions),
                                      0.0*np.ones(n_predictions),
                                      sa,
                                      ss,
                                      nb,
                                      sb])

        return self.return_prediction(*list(prediction_params)).astype('float32')

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
        mu_x : float
            x position
        mu_y : float
            y position
        prf_size : float
            sigma_1
        prf_amplitude : float
            Norm Param A
        bold_baseline : float
            BOLD baseline (generally kept fixed)
        neural_baseline : float
            Norm Param B
        srf_amplitude : float
            Norm Param C
        srf_size : float
            sigma_2
        surround_baseline : float
            Norm Param D
        hrf_1, hrf_2 : floats, optional
            hrf parameters, specified only if hrf is being fit to data, otherwise not needed.

        Returns
        -------
        numpy.ndarray
            prediction(s) given the model
        """

        if hrf_1 is not None and hrf_2 is not None:
            current_hrf = self.hrf.create_spm_hrf(
                force=True, TR=self.stimulus.TR, hrf_params=[1.0, hrf_1, hrf_2])

        assert self.hrf.hasValues(), "Initialize HRF values first!"
        current_hrf = self.hrf.values

        # create the rfs

        prf = np.rot90(gauss2D_iso_cart(x=self.stimulus.x_coordinates[..., np.newaxis],
                                        y=self.stimulus.y_coordinates[...,
                                                                      np.newaxis],
                                        mu=(mu_x, mu_y),
                                        sigma=prf_size,
                                        normalize_RFs=self.normalize_RFs).T, axes=(1, 2))

        # surround receptive field (denominator)
        srf = np.rot90(gauss2D_iso_cart(x=self.stimulus.x_coordinates[..., np.newaxis],
                                        y=self.stimulus.y_coordinates[...,
                                                                      np.newaxis],
                                        mu=(mu_x, mu_y),
                                        sigma=srf_size,
                                        normalize_RFs=self.normalize_RFs).T, axes=(1, 2))

        dm = self.stimulus.design_matrix

        # create normalization model timecourse
        neural_tc = (prf_amplitude[..., np.newaxis] * stimulus_through_prf(prf, dm, self.stimulus.dx) + neural_baseline[..., np.newaxis]) /\
            (srf_amplitude[..., np.newaxis] * stimulus_through_prf(srf, dm, self.stimulus.dx) + surround_baseline[..., np.newaxis]) \
            - neural_baseline[..., np.newaxis] / \
            surround_baseline[..., np.newaxis]

        tc = self.convolve_timecourse_hrf(neural_tc, current_hrf)

        if not self.filter_predictions:
            return bold_baseline[..., np.newaxis] + tc
        else:
            return bold_baseline[..., np.newaxis] + filter_predictions(
                tc,
                self.filter_type,
                self.filter_params)


class DoG_Iso2DGaussianModel(Iso2DGaussianModel):
    """redefining class for difference of Gaussians in iterative fit.
    """

    def create_grid_predictions(self,
                                gaussian_params,
                                sa,
                                ss):
        """create_predictions

        creates predictions for a given set of parameters

        [description]

        Parameters
        ----------
        gaussian_params: ndarray size (3)
            containing prf position and size.
        sa,ss: ndarrays
            containing the range of grid values for other DoG model parameters
            (surroud amplitude, surround size (sigma_2))


        """
        n_predictions = len(sa)

        prediction_params = np.array([gaussian_params[0]*np.ones(n_predictions),
                                      gaussian_params[1] *
                                      np.ones(n_predictions),
                                      gaussian_params[2] *
                                      np.ones(n_predictions),
                                      1.0*np.ones(n_predictions),
                                      0.0*np.ones(n_predictions),
                                      sa,
                                      ss])

        return self.return_prediction(*list(prediction_params)).astype('float32')

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
        if hrf_1 is not None and hrf_2 is not None:
            current_hrf = self.hrf.create_spm_hrf(
                force=True, TR=self.stimulus.TR, hrf_params=[1.0, hrf_1, hrf_2])

        assert self.hrf.hasValues(), "Initialize HRF values first!"
        current_hrf = self.hrf.values

        # create the rfs
        prf = np.rot90(gauss2D_iso_cart(x=self.stimulus.x_coordinates[..., np.newaxis],
                                        y=self.stimulus.y_coordinates[...,
                                                                      np.newaxis],
                                        mu=(mu_x, mu_y),
                                        sigma=prf_size,
                                        normalize_RFs=self.normalize_RFs).T, axes=(1, 2))

        # surround receptive field
        srf = np.rot90(gauss2D_iso_cart(x=self.stimulus.x_coordinates[..., np.newaxis],
                                        y=self.stimulus.y_coordinates[...,
                                                                      np.newaxis],
                                        mu=(mu_x, mu_y),
                                        sigma=srf_size,
                                        normalize_RFs=self.normalize_RFs).T, axes=(1, 2))

        dm = self.stimulus.design_matrix

        neural_tc = prf_amplitude[..., np.newaxis] * stimulus_through_prf(prf, dm, self.stimulus.dx) - \
            srf_amplitude[..., np.newaxis] * \
            stimulus_through_prf(srf, dm, self.stimulus.dx)

        tc = self.convolve_timecourse_hrf(neural_tc, current_hrf)

        if not self.filter_predictions:
            return bold_baseline[..., np.newaxis] + tc
        else:
            return bold_baseline[..., np.newaxis] + filter_predictions(
                tc,
                self.filter_type,
                self.filter_params)

class STDN_Iso2DGaussianModel(Iso2DGaussianModel):
    """STDN_Iso2DGaussianModel
    
    Redefining class for spatio-temporal normalization model

    """

    def __init__(self, 
                 stimulus,
                 fsample,
                 hrf: HRF = None,
                 filter_predictions=False,
                 filter_type='dc',
                 filter_params={},
                 normalize_RFs=False,
                 **kwargs):
        """__init__ for STDN_Iso2DGaussianModel

            constructor, sets up stimulus and hrf for this Model

            Parameters
            ----------
            stimulus : PRFStimulus2D
                Stimulus object specifying the information about the stimulus,
                and the space in which it lives.
            fsample : float, int
                Sampling frequency of data in Hz.
            hrf : HRF, array-like, None
                HRF for this Model.
            filter_predictions : boolean, optional
                whether to high-pass filter the predictions, default False
            filter_type, filter_params : see timecourse.py
            normalize_RFs : whether or not to normalize the RF volumes (generally not needed).
            """
        self.fsample = fsample
        super().__init__(stimulus=stimulus, hrf=hrf, filter_predictions=filter_predictions, filter_type=filter_type, filter_params=filter_params, normalize_RFs=normalize_RFs, **kwargs)

    def create_grid_predictions(self, gaussian_params, norm_params, irf_shape, irf_weight, neural_decay, adaptation_decay):
        """create_predictions

        creates predictions for a given set of parameters

        [description]

        Parameters
        ----------
        gaussian_params: ndarray size (3)
            containing prf position and size.
        sa,ss,nb,sb: ndarrays
            containing the range of grid values for other norm model parameters
            (surroud amplitude (C), surround size (sigma_2), neural baseline (B), surround baseline (D))


        """
        n_predictions = len(irf_shape)

        prediction_params = np.array([gaussian_params[0]*np.ones(n_predictions),
                                      gaussian_params[1] *
                                      np.ones(n_predictions),
                                      gaussian_params[2] *
                                      np.ones(n_predictions),
                                      1.0*np.ones(n_predictions),
                                      0.0*np.ones(n_predictions),
                                      norm_params[0] *
                                      np.ones(n_predictions),
                                      norm_params[1] *
                                      np.ones(n_predictions),
                                      norm_params[2] *
                                      np.ones(n_predictions),
                                      norm_params[3] *
                                      np.ones(n_predictions),
                                      irf_shape,
                                      irf_weight,
                                      neural_decay,
                                      adaptation_decay])

        return self.return_prediction(*list(prediction_params)).astype('float32')

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
                          irf_shape,
                          irf_weight,
                          neural_decay,
                          decay_amplitude,
                          adaptation_decay,
                          adaptation_amplitude,
                          hrf_1=None,
                          hrf_2=None,
                          ):
        """return_prediction [summary]

        returns the prediction for a single set of parameters.

        Parameters
        ----------
        mu_x : float
            x position
        mu_y : float
            y position
        prf_size : float
            sigma_1
        prf_amplitude : float
            Norm Param A
        bold_baseline : float
            BOLD baseline (generally kept fixed)
        neural_baseline : float
            Norm Param B
        srf_amplitude : float
            Norm Param C
        srf_size : float
            sigma_2
        surround_baseline : float
            Norm Param D
        irf_shape : float
            STDN param tau 1
        irf_weight : float
            STDN param w (weight of second, negative gamma function)
        neural_decay : float
            STDN param tau 2 (time constant of exponential decay for activation pool)
        adaptation_decay : float
            STDN param tau 3 (time constant of exponential decay for normalization pool)
        hrf_1, hrf_2 : floats, optional
            hrf parameters, specified only if hrf is being fit to data, otherwise not needed.

        Returns
        -------
        numpy.ndarray
            prediction(s) given the model
        """

        if hrf_1 is not None and hrf_2 is not None:
            current_hrf = self.hrf.create_spm_hrf(
                force=True, TR=self.stimulus.TR, hrf_params=[1.0, hrf_1, hrf_2])

        assert self.hrf.hasValues(), "Initialize HRF values first!"
        current_hrf = self.hrf.values

        # create the rfs

        prf = np.rot90(gauss2D_iso_cart(x=self.stimulus.x_coordinates[..., np.newaxis],
                                        y=self.stimulus.y_coordinates[...,
                                                                      np.newaxis],
                                        mu=(mu_x, mu_y),
                                        sigma=prf_size,
                                        normalize_RFs=self.normalize_RFs).T, axes=(1, 2))

        # surround receptive field (denominator)
        srf = np.rot90(gauss2D_iso_cart(x=self.stimulus.x_coordinates[..., np.newaxis],
                                        y=self.stimulus.y_coordinates[...,
                                                                      np.newaxis],
                                        mu=(mu_x, mu_y),
                                        sigma=srf_size,
                                        normalize_RFs=self.normalize_RFs).T, axes=(1, 2))

        dm = self.stimulus.design_matrix

        # timepoints = np.arange(dm.shape[2])
        # timepoints = np.linspace(0, dm.shape[2] / fsample, dm.shape[2])
        timepoints = np.array(range(dm.shape[2])) / self.fsample

        # create normalization model timecourse

        def gamma_function(t, tau):
            return np.exp(-t / tau)

        def stimulus_through_irf(prf, dm, dx, t, tau, weight, mask=None):
            spatial_tc = stimulus_through_prf(prf, dm, dx, mask=mask)
            difference_of_gammas = t * gamma_function(t, tau) - weight * t * gamma_function(t, 1.5 * tau)
            if difference_of_gammas.ndim < spatial_tc.ndim:
                difference_of_gammas = np.reshape(difference_of_gammas, (1, difference_of_gammas.shape[0]))

            # # normalize by peak (1)
            difference_of_gammas = difference_of_gammas / difference_of_gammas.max()
            # normalize by length
            # difference_of_gammas = difference_of_gammas / np.linalg.norm(difference_of_gammas)

            # scipy fftconvolve does not have padding options so doing it manually
            pad_length = 20
            pad = np.tile(spatial_tc[:, 0], (pad_length, 1)).T
            padded_tc = np.hstack((pad, spatial_tc))
            spatio_temporal_tc = signal.fftconvolve(padded_tc, difference_of_gammas, axes=(-1))[..., pad_length:spatial_tc.shape[-1]+pad_length]
            return spatio_temporal_tc

        def exponential_decay(tc, timepoints, decay):
            pad_length = 20
            pad = np.tile(tc[:, 0], (pad_length, 1)).T
            padded_tc = np.hstack((pad, tc))

            gamma = gamma_function(t=timepoints, tau=decay)
            if gamma.ndim < padded_tc.ndim:
                gamma = np.reshape(gamma, (1, gamma.shape[0]))

            # # normalize by peak (1)
            # gamma = gamma / gamma.max()
            # normalize by length
            gamma = gamma / np.linalg.norm(gamma)

            convolved_tc = signal.fftconvolve(padded_tc, gamma, axes=(-1))[..., pad_length:tc.shape[-1]+pad_length]

            return convolved_tc

        spatio_temporal_tc = stimulus_through_irf(prf=prf, dm=dm, dx=self.stimulus.dx, t=timepoints, 
                                                    tau=irf_shape[..., np.newaxis], weight=irf_weight[..., np.newaxis])
        norm_spatio_temporal_tc = stimulus_through_irf(prf=srf, dm=dm, dx=self.stimulus.dx, t=timepoints, 
                                                    tau=irf_shape[..., np.newaxis], weight=irf_weight[..., np.newaxis])


        activation_pool = prf_amplitude[..., np.newaxis] * np.abs(spatio_temporal_tc) + neural_baseline[..., np.newaxis]
        normalization_pool = srf_amplitude[..., np.newaxis] * np.abs(norm_spatio_temporal_tc) + surround_baseline[..., np.newaxis]


        activation_decay = decay_amplitude[..., np.newaxis] * np.abs(exponential_decay(tc=spatio_temporal_tc, timepoints=timepoints, decay=neural_decay[..., np.newaxis]))
        normalization_decay = adaptation_amplitude[..., np.newaxis] * np.abs(exponential_decay(tc=norm_spatio_temporal_tc, timepoints=timepoints, decay=adaptation_decay[..., np.newaxis]))

        neural_tc = activation_pool / (normalization_pool + activation_decay + normalization_decay) - (neural_baseline[..., np.newaxis] / surround_baseline[..., np.newaxis])

        tc = self.convolve_timecourse_hrf(neural_tc, current_hrf)

        # if not self.filter_predictions:
        #     return bold_baseline[..., np.newaxis] + tc
        # else:
        #     return bold_baseline[..., np.newaxis] + filter_predictions(
        #         tc,
        #         self.filter_type,
        #         self.filter_params)

        simulations = {
            'spatio_temporal_tc': spatio_temporal_tc,
            'norm_spatio_temporal_tc': norm_spatio_temporal_tc,
            'activation_pool': activation_pool,
            'normalization_pool': normalization_pool,
            'activation_decay': activation_decay,
            'normalization_decay': normalization_decay,
            'neural_tc': neural_tc,
        }

        return simulations

class CFGaussianModel():

    """A class for constructing gaussian connective field models.
    """

    def __init__(self, stimulus):
        """__init__

        Parameters
        ----------
        stimulus: A CFstimulus object.
        """

        self.stimulus = stimulus

    def create_rfs(self):
        """create_rfs

        creates rfs for the grid search

        Returns
        ----------
        grid_rfs: The receptive field profiles for the grid. 
        vert_centres_flat: A vector that defines the vertex centre associated with each rf profile.
        sigmas_flat: A vector that defines the CF size associated with each rf profile.


        """

        assert hasattr(
            self, 'sigmas'), "please define a grid of CF sizes first."

        if self.func == 'cart':

            # Make the receptive fields extend over the distances controlled by each of the sigma.
            self.grid_rfs = np.array(
                [gauss1D_cart(self.stimulus.distance_matrix, 0, s) for s in self.sigmas])

        # Reshape.
        self.grid_rfs = self.grid_rfs.reshape(-1, self.grid_rfs.shape[-1])

        # Flatten out the variables that define the centres and the sigmas.
        self.vert_centres_flat = np.tile(
            self.stimulus.subsurface_verts, self.sigmas.shape)
        self.sigmas_flat = np.repeat(
            self.sigmas, self.stimulus.distance_matrix.shape[0])

    def stimulus_times_prfs(self):
        """stimulus_times_prfs

        creates timecourses for each of the rfs in self.grid_rfs

         Returns
        ----------
        predictions: The predicted timecourse that is the dot product of the data in the source subsurface and each rf profile.

        """

        assert hasattr(self, 'grid_rfs'), "please create the rfs first"
        self.predictions = stimulus_through_prf(
            self.grid_rfs, self.stimulus.design_matrix,
            1)

    def create_grid_predictions(self, sigmas, func='cart'):
        """Creates the grid rfs and rf predictions
        """

        self.sigmas = sigmas
        self.func = func
        self.create_rfs()
        self.stimulus_times_prfs()

    def return_prediction(self, sigma, beta, baseline, vert):
        """return_prediction

        Creates a prediction given a sigma, beta, baseline and vertex centre.

        Returns
        ----------

        A prediction for this parameter combination. 

        """

        beta = np.array(beta)
        baseline = np.array(baseline)

        # Find the row of the distance matrix that corresponds to that vertex.
        idx = np.where(self.stimulus.subsurface_verts == vert)[0][0]

        # We can grab the row of the distance matrix corresponding to this vertex and make the rf.
        rf = np.array(
            [gauss1D_cart(self.stimulus.distance_matrix[idx], 0, sigma)])

        # Dot with the data to make the predictions.
        neural_tc = stimulus_through_prf(rf, self.stimulus.design_matrix, 1)

        return baseline[..., np.newaxis] + beta[..., np.newaxis] * neural_tc
