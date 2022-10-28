import numpy as np
import scipy.signal as signal
from nilearn.glm.first_level.hemodynamic_models import spm_hrf, spm_time_derivative, spm_dispersion_derivative

from .rf import gauss2D_iso_cart, gauss1D_cart # import required RF shapes
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

    def create_hrf(self, hrf_params):
        """
        
        construct single or multiple HRFs        

        Parameters
        ----------
        hrf_params : TYPE, optional
            DESCRIPTION. The default is [1.0, 1.0, 0.0].

        Returns
        -------
        hrf : ndarray
            the hrf.

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
        tc : ndarray, 1D or 2D
            The timecourse(s) to be convolved.
        hrf : ndarray, 1D or 2D
            The HRF. Can be single, or a different one for each timecourse.

        Returns
        -------
        convolved_tc : ndarray
            Convolved timecourse.

        """
        #scipy fftconvolve does not have padding options so doing it manually
        pad_length = 20
        pad = np.tile(tc[:,0], (pad_length,1)).T
        padded_tc = np.hstack((pad,tc))
        
        
        if hrf.shape[0]>1:           
            assert hrf.shape[0] == tc.shape[0], f"{hrf.shape[0]} HRFs provided vs {tc.shape[0]} timecourses"
            median_hrf = np.median(hrf, axis=0).reshape(1,-1)
            if np.all([np.allclose(median_hrf, single_hrf.reshape(1,-1)) for single_hrf in hrf]):             
                
                convolved_tc = signal.fftconvolve(padded_tc, median_hrf, axes=(-1))[..., pad_length:tc.shape[-1]+pad_length]
                
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


class Iso2DGaussianModel(Model):
    """Iso2DGaussianModel
    To extend please create a setup_XXX_grid function for any new way of
    defining grids.
    """

    def __init__(self,
                 stimulus,
                 hrf=[1.0, 1.0, 0.0],
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
        hrf : string, list or numpy.ndarray, optional
            HRF shape for this Model.
            Can be 'direct', which implements nothing (for eCoG or later convolution),
            a list or array of 3, which are multiplied with the three spm HRF basis functions,
            and an array already sampled on the TR by the user.
            (the default is None, which implements standard spm HRF)
        filter_predictions : boolean, optional
            whether to high-pass filter the predictions, default False
        filter_type, filter_params : see timecourse.py
        normalize_RFs : whether or not to normalize the RF volumes (generally not needed).
        """
        super().__init__(stimulus)
        self.__dict__.update(kwargs)

        # HRF stuff
        if isinstance(hrf, str):
            if hrf == 'direct':  # for use with anything like eCoG with instantaneous irf
                self.hrf = 'direct'
                self.stimulus.convolved_design_matrix = np.copy(stimulus.design_matrix)
            
        else:
            # some specific hrf with spm basis set
            if ((isinstance(hrf, list)) or (isinstance(hrf, np.ndarray))) and len(hrf) == 3:
                
                self.hrf_params = np.copy(hrf)
                
                if hrf[0] == 1: 
                    self.hrf = self.create_hrf(hrf_params=hrf)
                else:
                    print("WARNING: hrf[0] is not 1. this will confound it with amplitude\
                          parameters. consider setting it to 1 unless you are absolutely sure of what you are doing.\
                          this will also prevent you from fitting the HRF.")
                    self.hrf = self.create_hrf(hrf_params=hrf)
                    
            # some specific hrf already defined at the TR (!)
            # elif isinstance(hrf, np.ndarray) and len(hrf) > 3:
            elif isinstance(hrf, np.ndarray) and hrf.shape[0] == 1 and hrf.shape[1] > 3:
                self.hrf = np.copy(hrf)
        
        
            self.stimulus.convolved_design_matrix = convolve_stimulus_dm(
                stimulus.design_matrix, hrf=self.hrf)

        # filtering and other stuff
        self.filter_predictions = filter_predictions
        self.filter_type = filter_type
        
        #settings for filter
        self.filter_params = filter_params
        
        #adding stimulus parameters
        self.filter_params['task_lengths'] = self.stimulus.task_lengths
        self.filter_params['task_names'] = self.stimulus.task_names
        self.filter_params['late_iso_dict'] = self.stimulus.late_iso_dict
      
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

        if current_hrf == 'direct':
            tc = stimulus_through_prf(rf, dm, self.stimulus.dx)
        else:
            tc = self.convolve_timecourse_hrf(stimulus_through_prf(rf, dm, self.stimulus.dx), current_hrf)
        

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
                                    gaussian_params[1]*np.ones(n_predictions),
                                    gaussian_params[2]*np.ones(n_predictions)*
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
        
        if current_hrf == 'direct':
            tc = stimulus_through_prf(rf, dm, self.stimulus.dx)**n[..., np.newaxis]
        else:
            tc = self.convolve_timecourse_hrf(stimulus_through_prf(rf, dm, self.stimulus.dx)**n[..., np.newaxis], current_hrf)

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
                                    gaussian_params[1]*np.ones(n_predictions),
                                    gaussian_params[2]*np.ones(n_predictions),
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
        
        if current_hrf == 'direct':
            tc = (prf_amplitude[..., np.newaxis] * stimulus_through_prf(prf, dm, self.stimulus.dx) + neural_baseline[..., np.newaxis]) /\
            (srf_amplitude[..., np.newaxis] * stimulus_through_prf(srf, dm, self.stimulus.dx) + surround_baseline[..., np.newaxis]) \
                - neural_baseline[..., np.newaxis]/surround_baseline[..., np.newaxis]
        else:
            tc = self.convolve_timecourse_hrf((prf_amplitude[..., np.newaxis] * stimulus_through_prf(prf, dm, self.stimulus.dx) + neural_baseline[..., np.newaxis]) /\
            (srf_amplitude[..., np.newaxis] * stimulus_through_prf(srf, dm, self.stimulus.dx) + surround_baseline[..., np.newaxis]) \
                - neural_baseline[..., np.newaxis]/surround_baseline[..., np.newaxis]
                , current_hrf)        


                
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
                                    gaussian_params[1]*np.ones(n_predictions),
                                    gaussian_params[2]*np.ones(n_predictions),
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

        if current_hrf == 'direct':
            tc = prf_amplitude[..., np.newaxis] * stimulus_through_prf(prf, dm, self.stimulus.dx) - \
            srf_amplitude[..., np.newaxis] * stimulus_through_prf(srf, dm, self.stimulus.dx)
        else:
            tc = self.convolve_timecourse_hrf(prf_amplitude[..., np.newaxis] * stimulus_through_prf(prf, dm, self.stimulus.dx) - \
            srf_amplitude[..., np.newaxis] * stimulus_through_prf(srf, dm, self.stimulus.dx), current_hrf)



        if not self.filter_predictions:
            return bold_baseline[..., np.newaxis] + tc
        else:
            return bold_baseline[..., np.newaxis] + filter_predictions(
                tc,
                self.filter_type,
                self.filter_params)

        
        
class CFGaussianModel():
    
    """A class for constructing gaussian connective field models.
    """
    
    
    def __init__(self,stimulus):
        
        
        """__init__
        
        Parameters
        ----------
        stimulus: A CFstimulus object.
        """
        
        self.stimulus=stimulus
        
        
        
    def create_rfs(self):
        
        """create_rfs

        creates rfs for the grid search
        
        Returns
        ----------
        grid_rfs: The receptive field profiles for the grid. 
        vert_centres_flat: A vector that defines the vertex centre associated with each rf profile.
        sigmas_flat: A vector that defines the CF size associated with each rf profile.
        
        
        """
        
        assert hasattr(self, 'sigmas'), "please define a grid of CF sizes first."
        
        if self.func=='cart':
            
            # Make the receptive fields extend over the distances controlled by each of the sigma.
            self.grid_rfs  = np.array([gauss1D_cart(self.stimulus.distance_matrix, 0, s) for s in self.sigmas])
        
        # Reshape.
        self.grid_rfs=self.grid_rfs.reshape(-1, self.grid_rfs.shape[-1])
        
        # Flatten out the variables that define the centres and the sigmas.
        self.vert_centres_flat=np.tile(self.stimulus.subsurface_verts,self.sigmas.shape)
        self.sigmas_flat=np.repeat(self.sigmas,self.stimulus.distance_matrix.shape[0])
        
        
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
        
        
    def create_grid_predictions(self,sigmas,func='cart'):
        
        """Creates the grid rfs and rf predictions
        """
        
        self.sigmas=sigmas
        self.func=func
        self.create_rfs()
        self.stimulus_times_prfs()
        
        
        
    def return_prediction(self,sigma,beta,baseline,vert):
        
        """return_prediction

        Creates a prediction given a sigma, beta, baseline and vertex centre.
        
        Returns
        ----------
        
        A prediction for this parameter combination. 
        
        """
        
        
        beta=np.array(beta)
        baseline=np.array(baseline)
        
        # Find the row of the distance matrix that corresponds to that vertex.
        idx=np.where(self.stimulus.subsurface_verts==vert)[0][0]
            
        # We can grab the row of the distance matrix corresponding to this vertex and make the rf.
        rf=np.array([gauss1D_cart(self.stimulus.distance_matrix[idx], 0, sigma)])
            
        # Dot with the data to make the predictions. 
        neural_tc = stimulus_through_prf(rf, self.stimulus.design_matrix, 1)
    

        return baseline[..., np.newaxis] + beta[..., np.newaxis] * neural_tc
        
    
        
        