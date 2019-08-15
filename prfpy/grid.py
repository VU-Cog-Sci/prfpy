import numpy as np
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
                 task_lengths=None,
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
            self.hrf = np.array([hrf[0] * spm_hrf(tr=self.stimulus.TR,
                                                  oversampling=1, time_length=40),
                                 hrf[1] * spm_time_derivative(tr=self.stimulus.TR,
                                                              oversampling=1, time_length=40),
                                 hrf[2] * spm_dispersion_derivative(tr=self.stimulus.TR,
                                                                    oversampling=1, time_length=40)]).sum(axis=0)
        # some specific hrf already defined at the TR (!)
        elif isinstance(hrf, np.ndarray) and len(hrf) > 3:
            self.hrf = hrf

        self.convolved_design_matrix = convolve_stimulus_dm(
            stimulus.design_matrix, hrf=self.hrf)

        # filtering stuff
        self.filter_predictions = filter_predictions
        self.window_length = window_length
        self.polyorder = polyorder
        self.highpass = highpass
        self.add_mean = add_mean
        self.task_lengths = task_lengths

    def create_rfs(self):
        """create_rfs

        creates rfs for the grid

        """
        assert hasattr(self, 'xs'), "please set up the grid first"
        self.grid_rfs = gauss2D_iso_cart(
            x=self.stimulus.x_coordinates[..., np.newaxis],
            y=self.stimulus.y_coordinates[..., np.newaxis],
            mu=np.array([self.xs.ravel(), self.ys.ravel()]),
            sigma=self.sizes.ravel())
        # won't have to perform exponentiation if all ns are one (the default
        # value)
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
            self.predictions = sgfilter_predictions(self.predictions,
                                                    window_length=self.window_length,
                                                    polyorder=self.polyorder,
                                                    highpass=self.highpass,
                                                    add_mean=self.add_mean,
                                                    task_lengths=self.task_lengths)
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
                              sigma=size).T
        # won't have to perform exponentiation if n == 1
        if n == 1:

            # create timecourse
            tc = stimulus_through_prf(rf, self.convolved_design_matrix)[0, :]
        else:
            dm = self.stimulus.design_matrix
            neural_tc = stimulus_through_prf(rf, dm)**n

            tc = signal.convolve(neural_tc[0, :],
                                 self.hrf,
                                 mode='full')[:dm.shape[-1]].T

        if not self.filter_predictions:
            return baseline + beta * tc
        else:
            return baseline + beta * sgfilter_predictions(tc,
                                                          window_length=self.window_length,
                                                          polyorder=self.polyorder,
                                                          highpass=self.highpass,
                                                          add_mean=self.add_mean,
                                                          task_lengths=self.task_lengths).T

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


class Norm_Iso2DGaussianGridder(Iso2DGaussianGridder):
    """Norm_Iso2DGaussianGridder

    Redefining class to use normalization model in iterative fitting

    """

    def return_single_prediction(self,
                                 mu_x,
                                 mu_y,
                                 prf_size,
                                 prf_amplitude,
                                 bold_baseline,

                                 neural_baseline,
                                 srf_amplitude,
                                 srf_size,
                                 surround_baseline
                                 ):
        """return_single_prediction

        returns the prediction for a single set of parameters.

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

        # create the rfs
        # not sure why we need to take the transpose here but ok. following
        # parent method from Tomas
        if srf_amplitude == 0.0 and surround_baseline == 0.0:
            surround_baseline = 1e-3
            

        prf = gauss2D_iso_cart(x=self.stimulus.x_coordinates[..., np.newaxis],
                               y=self.stimulus.y_coordinates[..., np.newaxis],
                               mu=(mu_x, mu_y),
                               sigma=prf_size).T

        # surround receptive field (denominator)
        srf = gauss2D_iso_cart(x=self.stimulus.x_coordinates[..., np.newaxis],
                               y=self.stimulus.y_coordinates[..., np.newaxis],
                               mu=(mu_x, mu_y),
                               sigma=srf_size).T

        dm = self.stimulus.design_matrix

        # create normalization model timecourse
        neural_tc = (prf_amplitude * stimulus_through_prf(prf, dm) + neural_baseline) /\
            (srf_amplitude * stimulus_through_prf(srf, dm) + surround_baseline)

        tc = signal.convolve(neural_tc[0, :],
                             self.hrf,
                             mode='full')[:dm.shape[-1]].T

        # tc /= tc.max()
        if not self.filter_predictions:
            return bold_baseline + tc
        else:
            return bold_baseline + sgfilter_predictions(tc,
                                                        window_length=self.window_length,
                                                        polyorder=self.polyorder,
                                                        highpass=self.highpass,
                                                        add_mean=self.add_mean,
                                                        task_lengths=self.task_lengths).T

    def gradient_single_prediction(self,
                                   mu_x,
                                   mu_y,
                                   prf_size,
                                   prf_amplitude,
                                   bold_baseline,

                                   neural_baseline,
                                   srf_amplitude,
                                   srf_size,
                                   surround_baseline
                                   ):
        """gradient_single_prediction

        returns the prediction gradient for a single set of parameters.

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

        # create the rfs
        # not sure why we need to take the transpose here but ok. following
        # parent method from Tomas
        x_coord = self.stimulus.x_coordinates[..., np.newaxis]
        y_coord = self.stimulus.y_coordinates[..., np.newaxis]

        prf = gauss2D_iso_cart(x=x_coord,
                               y=y_coord,
                               mu=(mu_x, mu_y),
                               sigma=prf_size).T

        # surround receptive field (denominator)
        srf = gauss2D_iso_cart(x=x_coord,
                               y=y_coord,
                               mu=(mu_x, mu_y),
                               sigma=srf_size).T

        dm = self.stimulus.design_matrix

        gradient = np.zeros((9, dm.shape[-1]))

        # mu_x gradient
        gradient[0, :] = 2 * prf_amplitude * stimulus_through_prf((x_coord.T - mu_x) * prf, dm) /\
            (2 * prf_size**2 * (srf_amplitude * stimulus_through_prf(srf, dm) + surround_baseline)) -\
            (2 * (prf_amplitude * stimulus_through_prf((x_coord.T - mu_x) * prf, dm) + neural_baseline
                  ) * srf_amplitude * stimulus_through_prf(srf, dm)) /\
            (2 * srf_size**2 * (srf_amplitude *
                                stimulus_through_prf(srf, dm) + surround_baseline)**2)

        # mu_y gradient
        gradient[1, :] = 2 * prf_amplitude * stimulus_through_prf((y_coord.T - mu_y) * prf, dm) /\
            (2 * prf_size**2 * (srf_amplitude * stimulus_through_prf(srf, dm) + surround_baseline)) -\
            (2 * (prf_amplitude * stimulus_through_prf((y_coord.T - mu_y) * prf, dm) + neural_baseline
                  ) * srf_amplitude * stimulus_through_prf(srf, dm)) /\
            (2 * srf_size**2 * (srf_amplitude *
                                stimulus_through_prf(srf, dm) + surround_baseline)**2)

        # prf_size gradient
        gradient[2, :] = (prf_amplitude * stimulus_through_prf(((x_coord.T-mu_x)**2+(y_coord.T - mu_y)**2) * prf, dm)) /\
            (prf_size**3 * (srf_amplitude *
                            stimulus_through_prf(srf, dm) + surround_baseline))

        # prf_amplitude gradient
        gradient[3, :] = stimulus_through_prf(prf, dm) /\
            (srf_amplitude * stimulus_through_prf(srf, dm) + surround_baseline)

        # neural_baseline gradient
        gradient[5, :] = 1/(srf_amplitude *
                            stimulus_through_prf(srf, dm) + surround_baseline)

        # srf_amplitude gradient
        gradient[6, :] = -(prf_amplitude*stimulus_through_prf(prf, dm) + neural_baseline) *\
            stimulus_through_prf(srf, dm) /\
            (srf_amplitude * stimulus_through_prf(srf, dm)+surround_baseline)**2

        # srf_size gradient
        gradient[7, :] = srf_amplitude * stimulus_through_prf((-(x_coord.T-mu_x)**2-(y_coord.T - mu_y)**2) * srf, dm) *\
            (prf_amplitude*stimulus_through_prf(prf, dm) + neural_baseline) /\
            (srf_size**3 * (srf_amplitude *
                            stimulus_through_prf(srf, dm)+surround_baseline)**2)

        # surround_baseline gradient
        gradient[8, :] = - (prf_amplitude*stimulus_through_prf(prf, dm) + neural_baseline) /\
            (srf_amplitude * stimulus_through_prf(srf, dm)+surround_baseline)**2

        for i in range(gradient.shape[0]):
            gradient[i, :] = signal.convolve(gradient[i, :],
                                             self.hrf,
                                             mode='full')[:dm.shape[-1]]

        if not self.filter_predictions:
            # BOLD baseline is the only parameter outside HRF convolution and SG filter
            gradient[4, :] = np.ones(dm.shape[-1])
            return gradient
        else:
            gradient = sgfilter_predictions(gradient,
                                            window_length=self.window_length,
                                            polyorder=self.polyorder,
                                            highpass=self.highpass,
                                            add_mean=self.add_mean,
                                            task_lengths=self.task_lengths)

            # BOLD baseline is the only parameter outside HRF convolution and SG filter
            gradient[4, :] = np.ones(dm.shape[-1])
            return gradient


class DoG_Iso2DGaussianGridder(Iso2DGaussianGridder):
    """redefining class for difference of Gaussians in iterative fit.
    """

    def return_single_prediction(self,
                                 mu_x,
                                 mu_y,
                                 prf_size,
                                 prf_amplitude,
                                 bold_baseline,

                                 srf_amplitude,
                                 srf_size
                                 ):
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
        prf_size : float
            size of pRF


        Returns
        -------
        numpy.ndarray
            single prediction given the model
        """
        
        # create the rfs
        # not sure why we need to take the transpose here but ok. following
        # parent method from Tomas
        prf = gauss2D_iso_cart(x=self.stimulus.x_coordinates[..., np.newaxis],
                               y=self.stimulus.y_coordinates[..., np.newaxis],
                               mu=(mu_x, mu_y),
                               sigma=prf_size).T

        # surround receptive field
        srf = gauss2D_iso_cart(x=self.stimulus.x_coordinates[..., np.newaxis],
                               y=self.stimulus.y_coordinates[..., np.newaxis],
                               mu=(mu_x, mu_y),
                               sigma=srf_size).T

        tc = prf_amplitude*stimulus_through_prf(prf, self.convolved_design_matrix)[0, :]\
            - srf_amplitude * \
            stimulus_through_prf(srf, self.convolved_design_matrix)[0, :]

        if not self.filter_predictions:
            return bold_baseline + tc
        else:
            return bold_baseline + sgfilter_predictions(tc,
                                                        window_length=self.window_length,
                                                        polyorder=self.polyorder,
                                                        highpass=self.highpass,
                                                        add_mean=self.add_mean,
                                                        task_lengths=self.task_lengths).T
