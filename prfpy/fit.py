import numpy as np
from scipy.optimize import fmin_powell, minimize
from scipy.stats import zscore
from copy import deepcopy
from joblib import Parallel, delayed


def error_function(
        parameters,
        args,
        data,
        objective_function):
    """
    Parameters
    ----------
    parameters : list or ndarray
        A tuple of values representing a model setting.
    args : dictionary
        Extra arguments to `objective_function` beyond those in `parameters`.
    data : ndarray
       The actual, measured time-series against which the model is fit.
    objective_function : callable
        The objective function that takes `parameters` and `args` and
        produces a model time-series.

    Returns
    -------
    error : float
        The residual sum of squared errors between the prediction and data.
    """
    return np.nan_to_num(np.sum((data - objective_function(*list(parameters), **args))**2), nan=1e12)
    #return 1-np.nan_to_num(pearsonr(data,np.nan_to_num(objective_function(*list(parameters), **args)[0]))[0])


#(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)

#old after args
#xtol, ftol, verbose=True, bounds=None, constraints=None, method=None

def iterative_search(model, data, start_params, args, **kwargs):
    """iterative_search

    Generic minimization function called by iterative_fit.
    Do not call this directly. Use iterative_fit instead.

    [description]

    Parameters
    ----------
    model : Model
        Object that provides the predictions using its
        `return_prediction` method
    data : 1D numpy.ndarray
        the data to fit, same dimensions as are returned by
        Model's `return_prediction` method
    start_params : list or 1D numpy.ndarray
        initial values for the fit
    args : dictionary, arguments to model.return_prediction that
        are not optimized    
    **kwargs : additional keyword arguments will be passed to 
        scipy.optimize.minimize

    Returns
    -------
    2-tuple
        first element: parameter values
        second element: rsq value
    """

    output = minimize(error_function, start_params,
                    args=(args, data, model.return_prediction),
                    **kwargs)
    
    return np.nan_to_num(np.r_[output['x'], 1 - (output['fun'])/(len(data) * data.var())])




class Fitter:
    """Fitter

    Superclass for classes that implement the different fitting methods,
    for a given model. It contains 2D-data and leverages a Model object.

    data should be two-dimensional so that all bookkeeping with regard to voxels,
    electrodes, etc is done by the user. Generally, a Fitter class should implement
    both a `grid_fit` and an `iterative_fit` method to be run in sequence.

    """

    def __init__(self, data, model, n_jobs=1, **kwargs):
        """__init__ sets up data and model

        Parameters
        ----------
        data : numpy.ndarray, 2D
            input data. First dimension units, Second dimension time
        model : prfpy.Model
            Model object that provides the grid and iterative search
            predictions.
        n_jobs : int, optional
            number of jobs to use in parallelization (iterative search), by default 1

        """
        assert len(data.shape) == 2, \
            "input data should be two-dimensional, with first dimension units and second dimension time"     

            
        self.data = data.astype('float32')
        
        self.model = model
        self.n_jobs = n_jobs

        self.__dict__.update(kwargs)

        self.n_units = self.data.shape[0]
        self.n_timepoints = self.data.shape[-1]

        self.data_var = self.data.var(axis=-1)

    def iterative_fit(self,
                      rsq_threshold,
                      verbose=False,
                      starting_params=None,                   
                      args={},
                      **kwargs):
        """
        Generic function for iterative fitting. Does not need to be
        redefined for new models. It is sufficient to define
        `insert_new_model_params` or `grid_fit` in the new model Fitter class,
        or provide explicit `starting_params`
        (see Extend_Iso2DGaussianFitter for examples).


        Parameters
        ----------
        rsq_threshold : float
            Rsq threshold for iterative fitting. Must be between 0 and 1.
        verbose : boolean, optional
            Whether to print output. The default is False.
        starting_params : ndarray, optional
            Explicit start for iterative fit. The default is None.
        bounds : list of tuples, optional
            Bounds for parameter minimization. The default is None.
            if bounds are None, will use Powell optimizer
            if bounds are not None, will use LBFGSB or trust-constr
        args : dictionary, optional
            Further arguments passed to iterative_search. The default is {}.
        **kwargs : additional keyword arguments will be passed to 
            iterative_search, and from there to scipy.optimize.minimize

        Returns
        -------
        None.

        """

        self.__dict__.update(kwargs)
        
        #bounds need special handling for the unit-wise case
        self.bounds  = np.array(kwargs.pop('bounds', None))


        assert rsq_threshold>0, 'rsq_threshold must be >0!'

        if starting_params is None:
            assert hasattr(
                self, 'gridsearch_params'), 'First use self.grid_fit,\
            or provide explicit starting parameters!'

            self.starting_params = self.gridsearch_params

        else:
            self.starting_params = starting_params
        
        #allows unit-wise bounds. this can be used to keep certain parameters fixed to a predetermined unit-specific value, while fitting others.
        if len(self.bounds.shape) == 2:
            self.bounds = np.repeat(self.bounds[np.newaxis,...], self.starting_params.shape[0], axis=0)
            
        if not hasattr(self,'rsq_mask'):
            #use the grid or explicitly provided params to select voxels to fit
            self.rsq_mask = self.starting_params[:, -1] > rsq_threshold

        self.iterative_search_params = np.zeros_like(self.starting_params)

        if self.rsq_mask.sum()>0:
            if np.any(self.bounds) != None:
                #remove bounds from general kwargs so it can be handled
                
                iterative_search_params = Parallel(self.n_jobs, verbose=verbose)(
                    delayed(iterative_search)(self.model,
                                              data,
                                              start_params,
                                              args=args,
                                              bounds=curr_bounds,
                                              **kwargs)
                    for (data, start_params, curr_bounds) in zip(self.data[self.rsq_mask], self.starting_params[self.rsq_mask, :-1], self.bounds[self.rsq_mask]))
            else:
                iterative_search_params = Parallel(self.n_jobs, verbose=verbose)(
                    delayed(iterative_search)(self.model,
                                              data,
                                              start_params,
                                              args=args,
                                              **kwargs)
                    for (data, start_params) in zip(self.data[self.rsq_mask], self.starting_params[self.rsq_mask, :-1]))            
            
            self.iterative_search_params[self.rsq_mask] = np.array(
                iterative_search_params)
            
                
    def crossvalidate_fit(self,
                          test_data,
                          test_stimulus=None,
                          single_hrf=False):
        """
        Simple function to crossvalidate results of previous iterative fitting.
       

        Parameters
        ----------
        test_data : ndarray
            Test data for crossvalidation.
        test_stimulus : PRFStimulus, optional
            PRF stimulus for test. If same as train data, not needed.
        single_hrf : Bool
            If True, uses the average-across-units HRF params in crossvalidation

        Returns
        -------
        None.

        """

        assert hasattr(
                self, 'iterative_search_params'), 'First use self.iterative_fit,'      
        
        #to hande cases where test_data and fit_data have different stimuli
        if test_stimulus is not None:                
            fit_stimulus = deepcopy(self.model.stimulus)   
            
            self.model.stimulus = test_stimulus            
            self.model.filter_params['task_lengths'] = test_stimulus.task_lengths
            self.model.filter_params['task_names'] = test_stimulus.task_names
            self.model.filter_params['late_iso_dict'] = test_stimulus.late_iso_dict
            
        if self.rsq_mask.sum()>0:
            if single_hrf:
                median_hrf_params = np.median(self.iterative_search_params[self.rsq_mask,-3:-1],
                                               axis=0)
                
                self.iterative_search_params[self.rsq_mask,-3:-1] = median_hrf_params
                
                
            test_predictions = self.model.return_prediction(*list(self.iterative_search_params[self.rsq_mask,:-1].T))
            
            if test_stimulus is not None:
                self.model.stimulus = fit_stimulus
                self.model.filter_params['task_lengths'] = fit_stimulus.task_lengths
                self.model.filter_params['task_names'] = fit_stimulus.task_names
                self.model.filter_params['late_iso_dict'] = fit_stimulus.late_iso_dict  
                
            #calculate CV-rsq        
            CV_rsq = np.nan_to_num(1-np.sum((test_data[self.rsq_mask]-test_predictions)**2, axis=-1)/(test_data.shape[-1]*test_data[self.rsq_mask].var(-1)))

            self.iterative_search_params[self.rsq_mask,-1] = CV_rsq
        else:
            print("No voxels/vertices above Rsq threshold were found.")


        if self.data.shape == test_data.shape:
              
            self.noise_ceiling = np.zeros(self.n_units)
            
            n_c = 1-np.sum((test_data[self.rsq_mask]-self.data[self.rsq_mask])**2, axis=-1)/(test_data.shape[-1]*test_data[self.rsq_mask].var(-1))
            
            self.noise_ceiling[self.rsq_mask] = n_c

        
    


class Iso2DGaussianFitter(Fitter):
    """Iso2DGaussianFitter

    Class that implements the different fitting methods
    on a two-dimensional isotropic Gaussian pRF model,
    leveraging a Model object.

    """

    def grid_fit(self,
                 ecc_grid,
                 polar_grid,
                 size_grid,
                 verbose=False,
                 n_batches=1,
                 fixed_grid_baseline=None,
                 grid_bounds=None,
                 hrf_1_grid=None,
                 hrf_2_grid=None):
        """grid_fit

        performs grid fit using provided grids and predictor definitions


        Parameters
        ----------
        ecc_grid : 1D ndarray
            array of eccentricity values in grid
        polar_grid : 1D ndarray
            array of polar angle values in grid
        size_grid : 1D ndarray
            array of size values in grid
        verbose : boolean, optional
            print output. The default is False.
        n_batches : int, optional
            The data is split in n_batches of units and
            grid fit is performed in parallel.
        fixed_grid_baseline : float, optional
            The default is None. If not None, bold baseline will be fixed
            to this value (recommended).
        grid_bounds : list containing one tuple, optional
            The default is None. If not None, only values of pRF amplitude
            between grid_bounds[0][0] and grid_bounds[0][1] will be allowed.
            This is generally used to only allow positive pRFs, for example by
            specifying grid_bounds = [(0,1000)], only pRFs with amplitude
            between 0 and 1000 will be allowed in the grid fit  
        hrf_1_grid : 1D ndarray, optional
            The default is None. If not None, and if 
            self.use_previous_gaussian_fitter_hrf is False,
            will perform grid over these values of the hrf_1 parameter.
        hrf_1_grid : 1D ndarray, optional
            The default is None. If not None, and if 
            self.use_previous_gaussian_fitter_hrf is False,
            will perform grid over these values of the hrf_1 parameter.
        Returns
        -------
        None.

        """
        # setting up grid for norm model new params
        if hrf_1_grid is None or hrf_2_grid is None:
            eccs, polars, sizes = np.meshgrid(
            ecc_grid, polar_grid, size_grid)
            mu_x, mu_y = np.cos(polars) * eccs, np.sin(polars) * eccs

            self.hrf_1 = None
            self.hrf_2 = None
        else:
            eccs, polars, sizes, hrf_1, hrf_2 = np.meshgrid(
            ecc_grid, polar_grid, size_grid, hrf_1_grid, hrf_2_grid)
            mu_x, mu_y = np.cos(polars) * eccs, np.sin(polars) * eccs

            self.hrf_1 = hrf_1.ravel()
            self.hrf_2 = hrf_2.ravel()         

        self.mu_x = mu_x.ravel()
        self.mu_y = mu_y.ravel()
        self.sizes = sizes.ravel()

        self.n_predictions = len(self.mu_x)

        self.grid_predictions = self.model.create_grid_predictions(self.mu_x,self.mu_y,self.sizes,
                                                         self.hrf_1,self.hrf_2)

        # this function analytically computes best-fit rsq, slope, and baseline
        # for a given batch of units (faster than scipy/numpy lstsq).
        def rsq_betas_for_batch(data, vox_num, predictions,
                                n_timepoints, data_var,
                                sum_preds, square_norm_preds):
            result = np.zeros((data.shape[0], 4), dtype='float32')
            for vox_data, num, idx in zip(
                data, vox_num, np.arange(
                    data.shape[0])):
                # bookkeeping
                sumd = np.sum(vox_data)

                # best slopes and baselines for voxel for predictions
                if fixed_grid_baseline is None:
                    slopes = (n_timepoints * np.dot(vox_data, predictions.T) - sumd *
                              sum_preds) / (n_timepoints * square_norm_preds - sum_preds**2)
                    baselines = (sumd - slopes * sum_preds) / n_timepoints
                else:                    
                    slopes = (np.dot(vox_data-fixed_grid_baseline, predictions.T)) / (square_norm_preds)                   
                    baselines = fixed_grid_baseline * np.ones_like(slopes)

                # resid and rsq
                resid = np.linalg.norm((vox_data -
                                        slopes[..., np.newaxis] *
                                        predictions -
                                        baselines[..., np.newaxis]), axis=-
                                       1, ord=2)

                
                #enforcing a bound on the grid slope (i.e. prf amplitude)
                if grid_bounds is not None:
                    resid[slopes<grid_bounds[0][0]] = +np.inf
                    resid[slopes>grid_bounds[0][1]] = +np.inf
                    

                best_pred_voxel = np.nanargmin(resid)

                rsq = 1 - resid[best_pred_voxel]**2 / \
                    (n_timepoints * data_var[num])

                result[idx, :] = best_pred_voxel, rsq, baselines[best_pred_voxel], slopes[best_pred_voxel]

            return result

        # bookkeeping
        sum_preds = np.sum(self.grid_predictions, axis=-1)
        square_norm_preds = np.linalg.norm(
            self.grid_predictions, axis=-1, ord=2)**2

        # split data in batches
        split_indices = np.array_split(
            np.arange(self.data.shape[0]), n_batches)
        data_batches = np.array_split(self.data, n_batches, axis=0)
        if verbose:
            print("Each batch contains approx. " +
                  str(data_batches[0].shape[0]) + " voxels.")

        # perform grid fit
        grid_search_rbs = Parallel(self.n_jobs, verbose=verbose)(
            delayed(rsq_betas_for_batch)(
                data=data,
                vox_num=vox_num,
                predictions=self.grid_predictions,
                n_timepoints=self.n_timepoints,
                data_var=self.data_var,
                sum_preds=sum_preds,
                square_norm_preds=square_norm_preds)
            for data, vox_num in zip(data_batches, split_indices))

        grid_search_rbs = np.concatenate(grid_search_rbs, axis=0)

        max_rsqs = grid_search_rbs[:, 0].astype('int')
        self.gridsearch_r2 = grid_search_rbs[:, 1]
        self.best_fitting_baseline = grid_search_rbs[:, 2]
        self.best_fitting_beta = grid_search_rbs[:, 3]

        # output
        if hrf_1_grid is not None and hrf_2_grid is not None:
            self.gridsearch_params = np.array([
                self.mu_x[max_rsqs],
                self.mu_y[max_rsqs],
                self.sizes[max_rsqs],
                self.best_fitting_beta,
                self.best_fitting_baseline,
                self.hrf_1[max_rsqs],
                self.hrf_2[max_rsqs],
                self.gridsearch_r2
            ]).T
        else:
            self.gridsearch_params = np.array([
                self.mu_x[max_rsqs],
                self.mu_y[max_rsqs],
                self.sizes[max_rsqs],
                self.best_fitting_beta,
                self.best_fitting_baseline,
                self.model.hrf_params[1] * np.ones(self.n_units),
                self.model.hrf_params[2] * np.ones(self.n_units),
                self.gridsearch_r2
            ]).T            

       

class Extend_Iso2DGaussianFitter(Iso2DGaussianFitter):
    """

    Generic superclass to extend the Gaussian Fitter. If an existing
    Iso2DGaussianFitter object with iterative_search_params is provided, the
    prf position, size, and rsq parameters will be used for further minimizations.

    """

    def __init__(self, model, data, n_jobs=1,
                 previous_gaussian_fitter=None,
                 use_previous_gaussian_fitter_hrf=False):
        """

        Parameters
        ----------
        model : prfpy.Model
            Model object that provides the grid and iterative search
            predictions.
        data : numpy.ndarray, 2D
            input data. First dimension units, Second dimension time
        n_jobs : int, optional
            number of jobs to use in parallelization (iterative search), by default 1
        previous_gaussian_fitter : Iso2DGaussianFitter, optional
            The default is None. Must have iterative_search_params. 
        use_previous_gaussian_fitter_hrf : boolean, optional
            The default is False. if True, will use the HRF results from
            previous_gaussian_fitter during grid_fit of this model.

        Returns
        -------
        None.

        """

        if previous_gaussian_fitter is not None:
            if not hasattr(previous_gaussian_fitter,
                           'iterative_search_params'):
                print('Warning: gaussian iter fit not performed. Explicit\
                      starting parameters or grid params will be needed.')

            self.previous_gaussian_fitter = previous_gaussian_fitter
            self.use_previous_gaussian_fitter_hrf = use_previous_gaussian_fitter_hrf

        super().__init__(data, model, n_jobs=n_jobs)

    def insert_new_model_params(self, old_params):
        """
        Function to insert new model parameters starting values for iterfitting.
        To be redefined appropriately for each model (see below for examples).
        If `grid_fit` is defined and performed, `self.gridsearch_params` take
        precedence, and this function becomes unnecessary.

        Generally should not be used. grid_fit is preferable.

        Parameters
        ----------
        old_params : ndarray [n_units, 8]
            Previous Gaussian fitter parameters and rsq.

        Returns
        -------
        new_params : ndarray [n_units, number of new model parameters]
            Starting parameters for iterative fit.
            To be redefined appropriately for each model.

        """

        new_params = old_params
        return new_params

    def iterative_fit(self,
                      rsq_threshold,
                      verbose=False,
                      starting_params=None,
                      bounds=None,
                      args={},
                      constraints=None,
                      xtol=1e-4,
                      ftol=1e-4):
        """
        Iterative_fit for models building on top of the Gaussian. Does not need to be
        redefined for new models. It is sufficient to define either
        `insert_new_model_params` or `grid_fit`, in a new model Fitter class,
        or provide explicit `starting_params`.


        Parameters
        ----------
        rsq_threshold : float
            Rsq threshold for iterative fitting. Must be between 0 and 1.
        verbose : boolean, optional
            Whether to print output. The default is False.
        starting_params : ndarray of shape (n_units, n_params+1), optional
            Explicit start for minimization. The default is None.
        bounds : list of tuples of shape (n_params,2) or (n_units,n_params,2), optional
            Bounds for parameter minimization. The default is None.
        args : dictionary, optional
            Further arguments passed to iterative_search. The default is {}.
        constraints: list of scipy.optimize.LinearConstraints and/or
            scipy.optimize.NonLinearConstraints
            if constraints are not None, will use trust-constr optimizer
        xtol : float, optional
            if allowed by optimizer, parameter tolerance for termination of fitting
        ftol : float, optional
            if allowed by optimizer, objective function tolerance for termination of fitting

        Returns
        -------
        None.

        """

        if starting_params is None and not hasattr(
            self, 'gridsearch_params') and hasattr(
                self, 'previous_gaussian_fitter'):
            
            print("Warning: could not find gridsearch_params nor\
                  previous_gaussian_fitter. Using insert_new_model_params\
                  to specify starting values of new model params. Not recommended.")

            starting_params = self.insert_new_model_params(
                self.previous_gaussian_fitter.iterative_search_params)
            
            #fit exactly the same voxels/vertices as previous
            if hasattr(self.previous_gaussian_fitter, 'rsq_mask'):
                self.rsq_mask = self.previous_gaussian_fitter.rsq_mask
            else:
                self.rsq_mask = self.previous_gaussian_fitter.gridsearch_params[:,-1] > rsq_threshold


        super().iterative_fit(rsq_threshold=rsq_threshold,
                              verbose=verbose,
                              starting_params=starting_params,
                              bounds=bounds,
                              args=args,
                              constraints=constraints,
                              xtol=xtol,
                              ftol=ftol)


class CSS_Iso2DGaussianFitter(Extend_Iso2DGaussianFitter):
    """CSS_Iso2DGaussianFitter

    Compressive Spatial Summation model
    """

    def insert_new_model_params(self, old_params):
        """
        Parameters
        ----------
        old_params : ndarray [n_units, 8]
            Previous Gaussian fitter parameters and rsq.

        Returns
        -------
        new_params : ndarray [n_units, 9]
            Starting parameters and rsq for CSS iterative fit.

        """
        # insert CSS exponent
        new_params = np.insert(old_params, 5, 1.0, axis=-1)
        return new_params

    def grid_fit(self,
                 exponent_grid,
                 gaussian_params=None,
                 verbose=False,
                 n_batches=1,
                 rsq_threshold=0.05,
                 fixed_grid_baseline=None,
                 grid_bounds=None,
                 hrf_1_grid=None,
                 hrf_2_grid=None):
        """
        This function performs a grid_fit for the normalization model new parameters.
        The fit is parallel over batches of voxels, and separate predictions are
        made for each voxels based on its previously obtained Gaussian parameters (position and size).
        These can be provided explicitly in `gaussian_params`, or otherwise
        they are obtained from `previous_gaussian_fitter.iterative_search_params`


        Parameters
        ----------
        exponent_grid : 1D ndarray
            Array of exponent values.
        gaussian_params : ndarray [n_units, 4], optional
            The Gaussian parms [x position, y position, prf size, rsq] can be
            provided explicitly. If not, a previous_gaussian_fitter must be
            provided. The default is None.
        verbose : boolean, optional
            print output. The default is False.
        n_batches : int, optional
            The data is split in n_batches of units and
            grid fit is performed in parallel.
        rsq_threshold : float, optional
            rsq threshold for grid fitting. The default is 0.05.
        fixed_grid_baseline : float, optional
            The default is None. If not None, bold baseline will be fixed
            to this value (recommended).
        grid_bounds : list containing one tuple, optional
            The default is None. If not None, only values of pRF amplitude
            between grid_bounds[0][0] and grid_bounds[0][1] will be allowed.
            This is generally used to only allow positive pRFs, for example by
            specifying grid_bounds = [(0,1000)], only pRFs with amplitude
            between 0 and 1000 will be allowed in the grid fit  
        hrf_1_grid : 1D ndarray, optional
            The default is None. If not None, and if 
            self.use_previous_gaussian_fitter_hrf is False,
            will perform grid over these values of the hrf_1 parameter.
        hrf_1_grid : 1D ndarray, optional
            The default is None. If not None, and if 
            self.use_previous_gaussian_fitter_hrf is False,
            will perform grid over these values of the hrf_1 parameter.

        Raises
        ------
        ValueError
            Raised if there is no previous_gaussian_fitter or gaussian params.

        """

        # setting up grid for norm model new params
        if hrf_1_grid is None or hrf_2_grid is None:
            nn = exponent_grid

            self.hrf_1 = None
            self.hrf_2 = None
        else:
            nn, hrf_1, hrf_2 = np.meshgrid(exponent_grid,
            hrf_1_grid, hrf_2_grid)   

            self.hrf_1 = hrf_1.ravel()
            self.hrf_2 = hrf_2.ravel()      
        
        self.nn = nn.ravel()

        self.n_predictions = len(self.nn)

        if gaussian_params is not None and gaussian_params.shape == (
                self.n_units, 4):
            self.gaussian_params = gaussian_params.astype('float32')
            self.gridsearch_rsq_mask = self.gaussian_params[:, -1] > rsq_threshold
            
        elif hasattr(self, 'previous_gaussian_fitter'):
            starting_params_grid = self.previous_gaussian_fitter.iterative_search_params
            self.gaussian_params = np.concatenate(
                (starting_params_grid[:, :3], starting_params_grid[:, -1][..., np.newaxis]), axis=-1)
            
            if hasattr(self.previous_gaussian_fitter, 'rsq_mask'):
                self.gridsearch_rsq_mask = self.previous_gaussian_fitter.rsq_mask
            else:
                self.gridsearch_rsq_mask = self.previous_gaussian_fitter.gridsearch_params[:, -1] > rsq_threshold

            if self.use_previous_gaussian_fitter_hrf:
                print("Using HRF from previous gaussian iterative fit")
                self.hrf_1 = self.previous_gaussian_fitter.iterative_search_params[:, -3]
                self.hrf_2 = self.previous_gaussian_fitter.iterative_search_params[:, -2]            
        else:
            print('Please provide suitable [n_units, 4] gaussian_params,\
                  or previous_gaussian_fitter')
            raise ValueError

        

        # this function analytically computes best-fit rsq, slope, and baseline
        # for a given batch of units (faster than scipy/numpy lstsq).
        def rsq_betas_for_batch(data,
                                vox_nums,
                                n_predictions,
                                n_timepoints,
                                data_var,
                                nn, hrf_1, hrf_2,
                                gaussian_params):

            result = np.zeros((data.shape[0], 4), dtype='float32')

            for vox_data, vox_num, idx in zip(
                data, vox_nums, np.arange(
                    data.shape[0])):

                # let the model create the timecourses, per voxel, since the
                # gridding is over new parameters, while size and position
                # are obtained from previous Gaussian fit

                if self.use_previous_gaussian_fitter_hrf:
                    hrf_1_vx = hrf_1[vox_num] * np.ones(n_predictions)
                    hrf_2_vx = hrf_2[vox_num] * np.ones(n_predictions)
                else:
                    hrf_1_vx = hrf_1
                    hrf_2_vx = hrf_2
                
                css_resc_gp = np.copy(gaussian_params[vox_num, :-1])
                
                predictions = self.model.create_grid_predictions(
                    css_resc_gp, nn, hrf_1_vx, hrf_2_vx)
                # bookkeeping
                sum_preds = np.sum(predictions, axis=-1)
                square_norm_preds = np.linalg.norm(
                    predictions, axis=-1, ord=2)**2
                sumd = np.sum(vox_data)

                # best slopes and baselines for voxel for predictions
                if fixed_grid_baseline is None:
                    slopes = (n_timepoints * np.dot(vox_data, predictions.T) - sumd *
                              sum_preds) / (n_timepoints * square_norm_preds - sum_preds**2)
                    baselines = (sumd - slopes * sum_preds) / n_timepoints
                else:                    
                    slopes = (np.dot(vox_data-fixed_grid_baseline, predictions.T)) / (square_norm_preds)                   
                    baselines = fixed_grid_baseline * np.ones_like(slopes)

                # find best prediction and store relevant data
                resid = np.linalg.norm((vox_data -
                                        slopes[..., np.newaxis] *
                                        predictions -
                                        baselines[..., np.newaxis]), ord=2, axis=-
                                       1)

                        
                #enforcing a bound on the grid slope (i.e. prf amplitude)
                if grid_bounds is not None:
                    resid[slopes<grid_bounds[0][0]] = +np.inf
                    resid[slopes>grid_bounds[0][1]] = +np.inf

                best_pred_voxel = np.nanargmin(resid)

                rsq = 1 - resid[best_pred_voxel]**2 / \
                    (n_timepoints * data_var[vox_num])

                result[idx, :] = best_pred_voxel, rsq, baselines[best_pred_voxel], slopes[best_pred_voxel]

            return result

        # masking and splitting data
        split_indices = np.array_split(np.arange(self.data.shape[0])[
                                       self.gridsearch_rsq_mask], n_batches)
        data_batches = np.array_split(
            self.data[self.gridsearch_rsq_mask], n_batches, axis=0)

        if verbose:
            print("Each batch contains approx. " +
                  str(data_batches[0].shape[0]) + " voxels.")

        # parallel grid search over (sequential) batches of voxels
        grid_search_rbs = Parallel(self.n_jobs, verbose=11)(
            delayed(rsq_betas_for_batch)(
                data=data,
                vox_nums=vox_nums,
                n_predictions=self.n_predictions,
                n_timepoints=self.n_timepoints,
                data_var=self.data_var,
                nn=self.nn,
                hrf_1=self.hrf_1,hrf_2=self.hrf_2,
                gaussian_params=self.gaussian_params)
            for data, vox_nums in zip(data_batches, split_indices))

        grid_search_rbs = np.concatenate(grid_search_rbs, axis=0)

        # store results
        max_rsqs = grid_search_rbs[:, 0].astype('int')
        self.gridsearch_r2 = grid_search_rbs[:, 1]
        self.best_fitting_baseline = grid_search_rbs[:, 2]
        self.best_fitting_beta = grid_search_rbs[:, 3]

        self.gridsearch_params = np.zeros((self.n_units, 9))

        self.gridsearch_params[self.gridsearch_rsq_mask,:-3] = np.array([
            self.gaussian_params[self.gridsearch_rsq_mask, 0],
            self.gaussian_params[self.gridsearch_rsq_mask, 1],
            self.gaussian_params[self.gridsearch_rsq_mask, 2] * np.sqrt(self.nn[max_rsqs]),
            self.best_fitting_beta,
            self.best_fitting_baseline,
            self.nn[max_rsqs]]).T

        self.gridsearch_params[self.gridsearch_rsq_mask,-1] = self.gridsearch_r2             

        if self.use_previous_gaussian_fitter_hrf:
            self.gridsearch_params[self.gridsearch_rsq_mask,-3:-1] = np.array([
                self.hrf_1[self.gridsearch_rsq_mask],
                self.hrf_2[self.gridsearch_rsq_mask]]).T
        elif hrf_1_grid is not None and hrf_2_grid is not None:
            self.gridsearch_params[self.gridsearch_rsq_mask,-3:-1] = np.array([
                self.hrf_1[max_rsqs],
                self.hrf_2[max_rsqs]]).T
        else:
            self.gridsearch_params[self.gridsearch_rsq_mask,-3:-1] = np.array([
                self.model.hrf_params[1] * np.ones(self.gridsearch_rsq_mask.sum()),
                self.model.hrf_params[2] * np.ones(self.gridsearch_rsq_mask.sum())]).T

class DoG_Iso2DGaussianFitter(Extend_Iso2DGaussianFitter):
    """DoG_Iso2DGaussianFitter

    Difference of Gaussians model
    """

    def insert_new_model_params(self, old_params):
        """
        Parameters
        ----------
        old_params : ndarray [n_units, 8]
            Previous Gaussian fitter parameters and rsq.

        Returns
        -------
        new_params : ndarray [n_units, 10]
            Starting parameters and rsq for DoG iterative fit.

        """
        # surround amplitude
        new_params = np.insert(old_params, 5, 0.5*old_params[:,3], axis=-1)
        # surround size
        new_params = np.insert(
            new_params,
            6,
            1.5*old_params[:,2],
            axis=-1)

        return new_params

    def grid_fit(self,
                 surround_amplitude_grid,
                 surround_size_grid,
                 gaussian_params=None,
                 verbose=False,
                 n_batches=1,
                 rsq_threshold=0.05,
                 fixed_grid_baseline=None,
                 grid_bounds=None,
                 hrf_1_grid=None,
                 hrf_2_grid=None):
        """
        This function performs a grid_fit for the normalization model new parameters.
        The fit is parallel over batches of voxels, and separate predictions are
        made for each voxels based on its previously obtained Gaussian parameters (position and size).
        These can be provided explicitly in `gaussian_params`, or otherwise
        they are obtained from `previous_gaussian_fitter.iterative_search_params`


        Parameters
        ----------
        surround_amplitude_grid : 1D ndarray
            Array of surround amplitude values.
        surround_size_grid : 1D ndarray
            Array of surround size values (sigma_2).
        gaussian_params : ndarray [n_units, 4], optional
            The Gaussian parms [x position, y position, prf size, rsq] can be
            provided explicitly. If not, a previous_gaussian_fitter must be
            provided. The default is None.
        verbose : boolean, optional
            print output. The default is False.
        n_batches : int, optional
            Number of voxel batches. The default is 1.
        rsq_threshold : float, optional
            rsq threshold for grid fitting. The default is 0.05.
        fixed_grid_baseline : float, optional
            The default is None. If not None, bold baseline will be fixed
            to this value (recommended).
        grid_bounds : list containing one or two tuples, optional
            The default is None. If not None, only values of pRF amplitude
            between grid_bounds[0][0] and grid_bounds[0][1] will be allowed.
            This is generally used to only allow positive pRFs, for example by
            specifying grid_bounds = [(0,1000)], only pRFs with amplitude
            between 0 and 1000 will be allowed in the grid fit.
            If list contains two tuples, second tuple will give bounds on
            Surround Amplitude.
        hrf_1_grid : 1D ndarray, optional
            The default is None. If not None, and if 
            self.use_previous_gaussian_fitter_hrf is False,
            will perform grid over these values of the hrf_1 parameter.
        hrf_1_grid : 1D ndarray, optional
            The default is None. If not None, and if 
            self.use_previous_gaussian_fitter_hrf is False,
            will perform grid over these values of the hrf_1 parameter.

        Raises
        ------
        ValueError
            Raised if there is no previous_gaussian_fitter or gaussian params.

        """

        # setting up grid for norm model new params
        if hrf_1_grid is None or hrf_2_grid is None:
            sa, ss = np.meshgrid(
            surround_amplitude_grid, surround_size_grid)

            self.hrf_1 = None
            self.hrf_2 = None
        else:
            sa, ss, hrf_1, hrf_2 = np.meshgrid(
            surround_amplitude_grid, surround_size_grid,
            hrf_1_grid, hrf_2_grid)   

            self.hrf_1 = hrf_1.ravel()
            self.hrf_2 = hrf_2.ravel()         

        self.sa = sa.ravel()
        self.ss = ss.ravel()

        self.n_predictions = len(self.sa)

        if gaussian_params is not None and gaussian_params.shape == (
                self.n_units, 4):
            self.gaussian_params = gaussian_params.astype('float32')
            self.gridsearch_rsq_mask = self.gaussian_params[:, -1] > rsq_threshold
            
        elif hasattr(self, 'previous_gaussian_fitter'):
            starting_params_grid = self.previous_gaussian_fitter.iterative_search_params
            self.gaussian_params = np.concatenate(
                (starting_params_grid[:, :3], starting_params_grid[:, -1][..., np.newaxis]), axis=-1)
            
            if hasattr(self.previous_gaussian_fitter, 'rsq_mask'):
                self.gridsearch_rsq_mask = self.previous_gaussian_fitter.rsq_mask
            else:
                self.gridsearch_rsq_mask = self.previous_gaussian_fitter.gridsearch_params[:, -1] > rsq_threshold

            if self.use_previous_gaussian_fitter_hrf:
                print("Using HRF from previous gaussian iterative fit")
                self.hrf_1 = self.previous_gaussian_fitter.iterative_search_params[:, -3]
                self.hrf_2 = self.previous_gaussian_fitter.iterative_search_params[:, -2]

        else:
            print('Please provide suitable [n_units, 4] gaussian_params,\
                  or previous_gaussian_fitter')
            raise ValueError

        
        # this function analytically computes best-fit rsq, slope, and baseline
        # for a given batch of units (faster than scipy/numpy lstsq).
        def rsq_betas_for_batch(data,
                                vox_nums,
                                n_predictions,
                                n_timepoints,
                                data_var,
                                sa, ss, hrf_1, hrf_2,
                                gaussian_params):

            result = np.zeros((data.shape[0], 4), dtype='float32')

            for vox_data, vox_num, idx in zip(
                data, vox_nums, np.arange(
                    data.shape[0])):

                # let the model create the timecourses, per voxel, since the
                # gridding is over new parameters, while size and position
                # are obtained from previous Gaussian fit
                if self.use_previous_gaussian_fitter_hrf:
                    hrf_1_vx = hrf_1[vox_num] * np.ones(n_predictions)
                    hrf_2_vx = hrf_2[vox_num] * np.ones(n_predictions)
                else:
                    hrf_1_vx = hrf_1
                    hrf_2_vx = hrf_2

                predictions = self.model.create_grid_predictions(
                    gaussian_params[vox_num, :-1], sa, ss, hrf_1_vx, hrf_2_vx)
                # bookkeeping
                sum_preds = np.sum(predictions, axis=-1)
                square_norm_preds = np.linalg.norm(
                    predictions, axis=-1, ord=2)**2
                sumd = np.sum(vox_data)

                # best slopes and baselines for voxel for predictions
                if fixed_grid_baseline is None:
                    slopes = (n_timepoints * np.dot(vox_data, predictions.T) - sumd *
                              sum_preds) / (n_timepoints * square_norm_preds - sum_preds**2)
                    baselines = (sumd - slopes * sum_preds) / n_timepoints
                else:                    
                    slopes = (np.dot(vox_data-fixed_grid_baseline, predictions.T)) / (square_norm_preds)                   
                    baselines = fixed_grid_baseline * np.ones_like(slopes)

                # find best prediction and store relevant data
                resid = np.linalg.norm((vox_data -
                                        slopes[..., np.newaxis] *
                                        predictions -
                                        baselines[..., np.newaxis]), ord=2, axis=-
                                       1)

                #enforcing a bound on the grid slope (i.e. prf amplitude)
                if grid_bounds is not None:
                    #first bound amplitude
                    resid[slopes<grid_bounds[0][0]] = +np.inf
                    resid[slopes>grid_bounds[0][1]] = +np.inf
                    if len(grid_bounds)>1:
                        #second bound surround amplitude
                        resid[(sa*slopes)<grid_bounds[1][0]] = +np.inf
                        resid[(sa*slopes)>grid_bounds[1][1]] = +np.inf                    
                    

                best_pred_voxel = np.nanargmin(resid)

                rsq = 1 - resid[best_pred_voxel]**2 / \
                    (n_timepoints * data_var[vox_num])

                result[idx, :] = best_pred_voxel, rsq, baselines[best_pred_voxel], slopes[best_pred_voxel]

            return result

        # masking and splitting data
        split_indices = np.array_split(np.arange(self.data.shape[0])[
                                       self.gridsearch_rsq_mask], n_batches)
        data_batches = np.array_split(
            self.data[self.gridsearch_rsq_mask], n_batches, axis=0)

        if verbose:
            print("Each batch contains approx. " +
                  str(data_batches[0].shape[0]) + " voxels.")

        # parallel grid search over (sequential) batches of voxels
        grid_search_rbs = Parallel(self.n_jobs, verbose=11)(
            delayed(rsq_betas_for_batch)(
                data=data,
                vox_nums=vox_nums,
                n_predictions=self.n_predictions,
                n_timepoints=self.n_timepoints,
                data_var=self.data_var,
                sa=self.sa,
                ss=self.ss,
                hrf_1=self.hrf_1,hrf_2=self.hrf_2,
                gaussian_params=self.gaussian_params)
            for data, vox_nums in zip(data_batches, split_indices))

        grid_search_rbs = np.concatenate(grid_search_rbs, axis=0)

        # store results
        max_rsqs = grid_search_rbs[:, 0].astype('int')
        self.gridsearch_r2 = grid_search_rbs[:, 1]
        self.best_fitting_baseline = grid_search_rbs[:, 2]
        self.best_fitting_beta = grid_search_rbs[:, 3]

        self.gridsearch_params = np.zeros((self.n_units, 10))

        self.gridsearch_params[self.gridsearch_rsq_mask,:-3] = np.array([
            self.gaussian_params[self.gridsearch_rsq_mask, 0],
            self.gaussian_params[self.gridsearch_rsq_mask, 1],
            self.gaussian_params[self.gridsearch_rsq_mask, 2],
            self.best_fitting_beta,
            self.best_fitting_baseline,
            self.sa[max_rsqs] * self.best_fitting_beta,
            self.ss[max_rsqs]]).T

        self.gridsearch_params[self.gridsearch_rsq_mask,-1] = self.gridsearch_r2      

        if self.use_previous_gaussian_fitter_hrf:
            self.gridsearch_params[self.gridsearch_rsq_mask,-3:-1] = np.array([
                self.hrf_1[self.gridsearch_rsq_mask],
                self.hrf_2[self.gridsearch_rsq_mask]]).T
        elif hrf_1_grid is not None and hrf_2_grid is not None:
            self.gridsearch_params[self.gridsearch_rsq_mask,-3:-1] = np.array([
                self.hrf_1[max_rsqs],
                self.hrf_2[max_rsqs]]).T
        else:
            self.gridsearch_params[self.gridsearch_rsq_mask,-3:-1] = np.array([
                self.model.hrf_params[1] * np.ones(self.gridsearch_rsq_mask.sum()),
                self.model.hrf_params[2] * np.ones(self.gridsearch_rsq_mask.sum())]).T     


class Norm_Iso2DGaussianFitter(Extend_Iso2DGaussianFitter):
    """Norm_Iso2DGaussianFitter

    Divisive Normalization model

    """

    def insert_new_model_params(self, old_params):
        """
        Note: this function is generally unused since there is an
        efficient grid_fit for the normalization model (below)

        Parameters
        ----------
        old_params : ndarray [n_units, 8]
            Previous Gaussian fitter parameters and rsq.

        Returns
        -------
        new_params : ndarray [n_units, 12]
            Starting parameters and rsq for norm iterative fit.

        """
        # surround amplitude
        new_params = np.insert(old_params, 5, 0.0, axis=-1)
        # surround size
        new_params = np.insert(
            new_params,
            6,
            1.5*old_params[:,2],
            axis=-1)
        # neural baseline
        new_params = np.insert(new_params, 7, 0.0, axis=-1)
            # surround baseline
        new_params = np.insert(new_params, 8, 1.0, axis=-1)

        return new_params

    def grid_fit(self,
                 surround_amplitude_grid,
                 surround_size_grid,
                 neural_baseline_grid,
                 surround_baseline_grid,
                 gaussian_params=None,
                 verbose=False,
                 n_batches=1,
                 rsq_threshold=0.05,
                 fixed_grid_baseline=None,
                 grid_bounds=None,
                 hrf_1_grid=None,
                 hrf_2_grid=None,
                 ecc_grid=None,
                 polar_grid=None,
                 size_grid=None,
                 surround_size_as_proportion=False,
                 ecc_in_stim_range=False):
        """
        This function performs a grid_fit for the normalization model new parameters.
        The fit is parallel over batches of voxels, and separate predictions are
        made for each voxels based on its previously obtained Gaussian parameters (position and size).
        These can be provided explicitly in `gaussian_params`, or otherwise
        they are obtained from `previous_gaussian_fitter.iterative_search_params`


        Parameters
        ----------
        surround_amplitude_grid : 1D ndarray
            Array of surround amplitude values (Norm param C).
        surround_size_grid : 1D ndarray
            Array of surround size values (sigma_2).
        neural_baseline_grid : 1D ndarray
            Array of neural baseline values (Norm param B).
        surround_baseline_grid : 1D ndarray
            Array of surround baseline values (Norm param D).
        gaussian_params : ndarray [n_units, 4], optional
            The Gaussian parms [x position, y position, prf size, rsq] can be
            provided explicitly. If not, a previous_gaussian_fitter must be
            provided. The default is None.
        verbose : boolean, optional
            print output. The default is False.
        n_batches : int, optional
            The data is split in n_batches of units and
            grid fit is performed in parallel.
        rsq_threshold : float, optional
            rsq threshold for grid fitting. The default is 0.05.
        fixed_grid_baseline : float, optional
            The default is None. If not None, bold baseline will be fixed
            to this value (recommended).
        grid_bounds : list containing one tuple, optional
            The default is None. If not None, only values of pRF amplitude
            between grid_bounds[0][0] and grid_bounds[0][1] will be allowed.
            This is generally used to only allow positive pRFs, for example by
            specifying grid_bounds = [(0,1000)], only pRFs with amplitude
            between 0 and 1000 will be allowed in the grid fit.
            If list contains two tuples, second tuple will give bounds on
            Neural Baseline (Norm param B).
        hrf_1_grid : 1D ndarray, optional
            The default is None. If not None, and if 
            self.use_previous_gaussian_fitter_hrf is False,
            will perform grid over these values of the hrf_1 parameter.
        hrf_2_grid : 1D ndarray, optional
            The default is None. If not None, and if 
            self.use_previous_gaussian_fitter_hrf is False,
            will perform grid over these values of the hrf_2 parameter.
        ecc_grid : 1D ndarray, optional
            The default is None. If all ecc_grid, polar_grid, and size_grid
            are not None, will perform full grid for DN model.
        polar_grid : 1D ndarray, optional
            The default is None. If all ecc_grid, polar_grid, and size_grid
            are not None, will perform full grid for DN model.
        size_grid : 1D ndarray, optional
            The default is None. If all ecc_grid, polar_grid, and size_grid
            are not None, will perform full grid for DN model.
        surround_size_as_proportion : boolean, optional
            If True, interpret surround_size_grid as a factor multiplying
            the activation pRF size. The default is False.
        ecc_in_stim_range : boolean, optional
            If True, rescale eccentricity and size
            of pRFs outside of the screen
            in gaussian params to be in screen. The default is False.

        Raises
        ------
        ValueError
            Raised if there is no previous_gaussian_fitter or gaussian params.

        """

        # setting up grid for norm model new params
        if ecc_grid is None or polar_grid is None or size_grid is None:
            if hrf_1_grid is None or hrf_2_grid is None:
                sa, ss, nb, sb = np.meshgrid(
                surround_amplitude_grid, surround_size_grid,
                neural_baseline_grid, surround_baseline_grid)

                self.hrf_1 = None
                self.hrf_2 = None
            else:
                sa, ss, nb, sb, hrf_1, hrf_2 = np.meshgrid(
                surround_amplitude_grid, surround_size_grid,
                neural_baseline_grid, surround_baseline_grid,
                hrf_1_grid, hrf_2_grid)   

                self.hrf_1 = hrf_1.ravel()
                self.hrf_2 = hrf_2.ravel()         

            self.sa = sa.ravel()
            self.ss = ss.ravel()
            self.nb = nb.ravel()
            self.sb = sb.ravel()

            self.n_predictions = len(self.nb)

            #gaussian params case, either explicit or from previous fitter
            if gaussian_params is not None:
                self.gaussian_params = gaussian_params.astype('float32')

                if self.use_previous_gaussian_fitter_hrf:

                    print("Using HRF from explicitly provided gaussian_params")
                    self.hrf_1 = gaussian_params[:, -3].astype('float32')
                    self.hrf_2 = gaussian_params[:, -2].astype('float32')

                #back in the grid also for DN model, as gauss
                if ecc_in_stim_range:
                    max_ecc_scr = self.model.stimulus.screen_size_degrees/2.0
                    ecc_gauss = np.sqrt(self.gaussian_params[:, 0]**2 + self.gaussian_params[:, 1]**2)
                    resc_fctr = np.min([max_ecc_scr/ecc_gauss, np.ones_like(ecc_gauss)], axis=0)

                    self.gaussian_params[:, :3] *= resc_fctr[...,np.newaxis]

                self.gridsearch_rsq_mask = self.gaussian_params[:, -1] > rsq_threshold
                
            elif hasattr(self, 'previous_gaussian_fitter'):
                starting_params_grid = self.previous_gaussian_fitter.iterative_search_params
                self.gaussian_params = np.concatenate(
                    (starting_params_grid[:, :3], starting_params_grid[:, -1][..., np.newaxis]), axis=-1)
                
                #back in the grid also for DN model, as gauss
                if ecc_in_stim_range:
                    max_ecc_scr = self.model.stimulus.screen_size_degrees/2.0
                    ecc_gauss = np.sqrt(self.gaussian_params[:, 0]**2 + self.gaussian_params[:, 1]**2)
                    resc_fctr = np.min([max_ecc_scr/ecc_gauss, np.ones_like(ecc_gauss)], axis=0)

                    self.gaussian_params[:, :3] *= resc_fctr[...,np.newaxis]
                
                if hasattr(self.previous_gaussian_fitter, 'rsq_mask'):
                    self.gridsearch_rsq_mask = self.previous_gaussian_fitter.rsq_mask
                else:
                    self.gridsearch_rsq_mask = self.previous_gaussian_fitter.gridsearch_params[:, -1] > rsq_threshold

                if self.use_previous_gaussian_fitter_hrf:
                    print("Using HRF from previous gaussian iterative fit")
                    self.hrf_1 = self.previous_gaussian_fitter.iterative_search_params[:, -3]
                    self.hrf_2 = self.previous_gaussian_fitter.iterative_search_params[:, -2]
                
            else:
                print('Please provide suitable [n_units, 4] gaussian_params,\
                    or previous_gaussian_fitter')
                raise ValueError
        else:
            print("Performing full grid for DN model")
            if self.use_previous_gaussian_fitter_hrf:
                print('Will not be able to use voxel-wise hrf parameter from gauss model')
                self.use_previous_gaussian_fitter_hrf=False
            self.gaussian_params = None
            self.gridsearch_rsq_mask = np.ones(self.data.shape[0], dtype='bool')

            if hrf_1_grid is None or hrf_2_grid is None:
                eccs, polars, sizes, sa, ss, nb, sb = np.meshgrid(
                ecc_grid, polar_grid, size_grid,
                surround_amplitude_grid, surround_size_grid,
                neural_baseline_grid, surround_baseline_grid)

                mu_x, mu_y = np.cos(polars) * eccs, np.sin(polars) * eccs

                self.hrf_1 = None
                self.hrf_2 = None
            else:
                eccs, polars, sizes, sa, ss, nb, sb, hrf_1, hrf_2 = np.meshgrid(
                ecc_grid, polar_grid, size_grid,
                surround_amplitude_grid, surround_size_grid,
                neural_baseline_grid, surround_baseline_grid,
                hrf_1_grid, hrf_2_grid)   

                mu_x, mu_y = np.cos(polars) * eccs, np.sin(polars) * eccs

                self.hrf_1 = hrf_1.ravel()
                self.hrf_2 = hrf_2.ravel()         


            self.mu_x = mu_x.ravel()
            self.mu_y = mu_y.ravel()
            self.sizes = sizes.ravel()
            self.sa = sa.ravel()
            self.ss = ss.ravel()
            self.nb = nb.ravel()
            self.sb = sb.ravel()

            self.n_predictions = len(self.nb)

            if surround_size_as_proportion:
                self.ss *= self.sizes

            if self.n_predictions>50000:
                splits = self.n_jobs
                grid_predictions = Parallel(self.n_jobs, verbose=11)(
                    delayed(self.model.create_grid_predictions)(mxx,myx,sx,
                                                                sax,
                                                                ssx,
                                                                nbx,sbx,
                                                                hrf1x,hrf2x)
                for mxx,myx,sx,sax,ssx,nbx,sbx,hrf1x,hrf2x in
                zip(np.array_split(self.mu_x,splits),np.array_split(self.mu_y,splits),
                    np.array_split(self.sizes,splits),np.array_split(self.sa,splits),
                    np.array_split(self.ss,splits),np.array_split(self.nb,splits),
                    np.array_split(self.sb,splits),np.array_split(self.hrf_1,splits),
                    np.array_split(self.hrf_2,splits))
                )
                grid_predictions = np.concatenate(grid_predictions,axis=0)
            else:
                grid_predictions = self.model.create_grid_predictions(
                    self.mu_x, self.mu_y, self.sizes,
                    self.sa, 
                    self.ss,
                    self.nb, self.sb, 
                    self.hrf_1, self.hrf_2)
            print(f'{self.n_predictions} full grid predictions completed')
            # bookkeeping
            self.sum_preds = np.sum(grid_predictions, axis=-1)
            self.square_norm_preds = np.linalg.norm(
                grid_predictions, axis=-1, ord=2)**2
            
        
        # this function analytically computes best-fit rsq, slope, and baseline
        # for a given batch of units (faster than scipy/numpy lstsq).
        def rsq_betas_for_batch(data,
                                vox_nums,
                                n_predictions,
                                n_timepoints,
                                data_var,
                                sa, ss, nb, sb, hrf_1, hrf_2,
                                gaussian_params):

            result = np.zeros((data.shape[0], 4), dtype='float32')

            for vox_data, vox_num, idx in zip(
                data, vox_nums, np.arange(
                    data.shape[0])):

                # let the model create the timecourses, per voxel, since the
                # gridding is over new parameters, while size and position
                # are obtained from previous Gaussian fit
                if self.gaussian_params is not None:
                    if self.use_previous_gaussian_fitter_hrf:
                        hrf_1_vx = hrf_1[vox_num] * np.ones(n_predictions)
                        hrf_2_vx = hrf_2[vox_num] * np.ones(n_predictions)
                    else:
                        hrf_1_vx = hrf_1
                        hrf_2_vx = hrf_2


                    mu_x = gaussian_params[vox_num, 0] * np.ones(n_predictions)
                    mu_y = gaussian_params[vox_num, 1] * np.ones(n_predictions)
                    size = gaussian_params[vox_num, 2] * np.ones(n_predictions)

                    if surround_size_as_proportion:
                        surr_size = gaussian_params[vox_num, 2] * ss
                    else:
                        surr_size = ss

                    predictions = self.model.create_grid_predictions(
                        mu_x, mu_y, size, sa, surr_size, nb, sb, hrf_1_vx, hrf_2_vx)
                    
                    # bookkeeping
                    sum_preds = np.sum(predictions, axis=-1)
                    square_norm_preds = np.linalg.norm(
                        predictions, axis=-1, ord=2)**2
                    
                else:
                    predictions = grid_predictions
                    sum_preds = self.sum_preds
                    square_norm_preds = self.square_norm_preds

 
                sumd = np.sum(vox_data)

                # best possible slopes and baselines
                if fixed_grid_baseline is None:
                    slopes = (n_timepoints * np.dot(vox_data, predictions.T) - sumd *
                              sum_preds) / (n_timepoints * square_norm_preds - sum_preds**2)
                    baselines = (sumd - slopes * sum_preds) / n_timepoints
                else:                    
                    slopes = (np.dot(vox_data-fixed_grid_baseline, predictions.T)) / (square_norm_preds)                    
                    baselines = fixed_grid_baseline * np.ones_like(slopes)

                # find best prediction and store relevant data
                resid = np.linalg.norm((vox_data -
                                        slopes[..., np.newaxis] *
                                        predictions -
                                        baselines[..., np.newaxis]), ord=2, axis=-
                                       1)

                #enforcing a bound on the grid slope (i.e. prf amplitude)
                if grid_bounds is not None:
                    #first bound amplitude
                    resid[slopes<grid_bounds[0][0]] = +np.inf
                    resid[slopes>grid_bounds[0][1]] = +np.inf
                    if len(grid_bounds)>1:
                        #second bound neural baseline (norm param B)
                        resid[(nb*slopes)<grid_bounds[1][0]] = +np.inf
                        resid[(nb*slopes)>grid_bounds[1][1]] = +np.inf    

                best_pred_voxel = np.nanargmin(resid)

                rsq = 1 - resid[best_pred_voxel]**2 / \
                    (n_timepoints * data_var[vox_num])

                result[idx, :] = best_pred_voxel, rsq, baselines[best_pred_voxel], slopes[best_pred_voxel]

            return result

        # masking and splitting data
        split_indices = np.array_split(np.arange(self.data.shape[0])[
                                       self.gridsearch_rsq_mask], n_batches)
        data_batches = np.array_split(
            self.data[self.gridsearch_rsq_mask], n_batches, axis=0)

        if verbose:
            print("Each batch contains approx. " +
                  str(data_batches[0].shape[0]) + " voxels.")

        # parallel grid search over (sequential) batches of voxels
        grid_search_rbs = Parallel(self.n_jobs, verbose=11)(
            delayed(rsq_betas_for_batch)(
                data=data,
                vox_nums=vox_nums,
                n_predictions=self.n_predictions,
                n_timepoints=self.n_timepoints,
                data_var=self.data_var,
                sa=self.sa,
                ss=self.ss,
                nb=self.nb,
                sb=self.sb,
                hrf_1=self.hrf_1,hrf_2=self.hrf_2,
                gaussian_params=self.gaussian_params)
            for data, vox_nums in zip(data_batches, split_indices))

        grid_search_rbs = np.concatenate(grid_search_rbs, axis=0)

        # store results
        max_rsqs = grid_search_rbs[:, 0].astype('int')
        self.gridsearch_r2 = grid_search_rbs[:, 1]
        self.best_fitting_baseline = grid_search_rbs[:, 2]
        self.best_fitting_beta = grid_search_rbs[:, 3]

        self.gridsearch_params = np.zeros((self.n_units, 12))

        #NOTE: GLM also rescales b
        #NOTE: ss as proportion of individual size 

        if self.gaussian_params is not None:
            #gaussian params situation
            self.gridsearch_params[self.gridsearch_rsq_mask,:-3] = np.array([
                self.gaussian_params[self.gridsearch_rsq_mask, 0],
                self.gaussian_params[self.gridsearch_rsq_mask, 1],
                self.gaussian_params[self.gridsearch_rsq_mask, 2],
                self.best_fitting_beta,
                self.best_fitting_baseline,
                self.sa[max_rsqs],
                self.ss[max_rsqs],
                self.nb[max_rsqs] * self.best_fitting_beta,
                self.sb[max_rsqs]]).T
            #ss as proportion
            if surround_size_as_proportion:
                self.gridsearch_params[self.gridsearch_rsq_mask,6] *= self.gaussian_params[self.gridsearch_rsq_mask, 2]
        else:
            #full grid situation (self.ss already rescaled)
            self.gridsearch_params[self.gridsearch_rsq_mask,:-3] = np.array([
                self.mu_x[max_rsqs],
                self.mu_y[max_rsqs],
                self.sizes[max_rsqs],
                self.best_fitting_beta,
                self.best_fitting_baseline,
                self.sa[max_rsqs],              
                self.ss[max_rsqs],            
                self.nb[max_rsqs] * self.best_fitting_beta,
                self.sb[max_rsqs]]).T

        self.gridsearch_params[self.gridsearch_rsq_mask,-1] = self.gridsearch_r2             

        if self.use_previous_gaussian_fitter_hrf:
            self.gridsearch_params[self.gridsearch_rsq_mask,-3:-1] = np.array([
                self.hrf_1[self.gridsearch_rsq_mask],
                self.hrf_2[self.gridsearch_rsq_mask]]).T
        elif hrf_1_grid is not None and hrf_2_grid is not None:
            self.gridsearch_params[self.gridsearch_rsq_mask,-3:-1] = np.array([
                self.hrf_1[max_rsqs],
                self.hrf_2[max_rsqs]]).T
        else:
            self.gridsearch_params[self.gridsearch_rsq_mask,-3:-1] = np.array([
                self.model.hrf_params[1] * np.ones(self.gridsearch_rsq_mask.sum()),
                self.model.hrf_params[2] * np.ones(self.gridsearch_rsq_mask.sum())]).T


        
        
class CFFitter(Fitter):
    
    """CFFitter

    Class that implements the different fitting methods
    on a gaussian CF model,
    leveraging a Model object.

    """
    
    
    def grid_fit(self,sigma_grid,verbose=False,n_batches=1000):
        
        
        """grid_fit

        performs grid fit using provided grids and predictor definitions


        Parameters
        ----------
        sigma_grid : 1D ndarray
            to be filled in by user
        verbose : boolean, optional
            print output. The default is False.
        n_batches : int, optional
            The grid fit is performed in parallel over n_batches of units.
            Batch parallelization is faster than single-unit
            parallelization and of sequential computing.

        Returns
        -------
        gridsearch_params: An array containing the gridsearch parameters.
        vertex_centres: An array containing the vertex centres.
        vertex_centres_dict: A dictionary containing the vertex centres.

        """
        
        
        self.model.create_grid_predictions(sigma_grid)
        self.model.predictions = self.model.predictions.astype('float32')
        
        def rsq_betas_for_batch(data, vox_num, predictions,
                                n_timepoints, data_var,
                                sum_preds, square_norm_preds):
            result = np.zeros((data.shape[0], 4), dtype='float32')
            for vox_data, num, idx in zip(
                data, vox_num, np.arange(
                    data.shape[0])):
                # bookkeeping
                sumd = np.sum(vox_data)

                # best slopes and baselines for voxel for predictions
                slopes = (n_timepoints * np.dot(vox_data, predictions.T) - sumd *
                          sum_preds) / (n_timepoints * square_norm_preds - sum_preds**2)
                baselines = (sumd - slopes * sum_preds) / n_timepoints

                # resid and rsq
                resid = np.linalg.norm((vox_data -
                                        slopes[..., np.newaxis] *
                                        predictions -
                                        baselines[..., np.newaxis]), axis=-
                                       1, ord=2)

                #to enforce, if possible, positive prf amplitude
                #if pos_prfs_only:
                #    if np.any(slopes>0):
                #        resid[slopes<=0] = +np.inf

                best_pred_voxel = np.nanargmin(resid)

                rsq = 1 - resid[best_pred_voxel]**2 / \
                    (n_timepoints * data_var[num])

                result[idx, :] = best_pred_voxel, rsq, baselines[best_pred_voxel], slopes[best_pred_voxel]

            return result

        # bookkeeping
        sum_preds = np.sum(self.model.predictions, axis=-1)
        square_norm_preds = np.linalg.norm(
            self.model.predictions, axis=-1, ord=2)**2

        # split data in batches
        split_indices = np.array_split(
            np.arange(self.data.shape[0]), n_batches)
        data_batches = np.array_split(self.data, n_batches, axis=0)
        if verbose:
            print("Each batch contains approx. " +
                  str(data_batches[0].shape[0]) + " voxels.")

        # perform grid fit
        grid_search_rbs = Parallel(self.n_jobs, verbose=verbose)(
            delayed(rsq_betas_for_batch)(
                data=data,
                vox_num=vox_num,
                predictions=self.model.predictions,
                n_timepoints=self.n_timepoints,
                data_var=self.data_var,
                sum_preds=sum_preds,
                square_norm_preds=square_norm_preds)
            for data, vox_num in zip(data_batches, split_indices))

        grid_search_rbs = np.concatenate(grid_search_rbs, axis=0)

        max_rsqs = grid_search_rbs[:, 0].astype('int')
        self.gridsearch_r2 = grid_search_rbs[:, 1]
        self.best_fitting_baseline = grid_search_rbs[:, 2]
        self.best_fitting_beta = grid_search_rbs[:, 3]

        # output
        self.gridsearch_params = np.array([
            self.model.vert_centres_flat[max_rsqs],
            self.model.sigmas_flat[max_rsqs],
            self.best_fitting_beta,
            self.best_fitting_baseline,
            self.gridsearch_r2
        ]).T
        
        # Put the vertex centres into a dictionary.
        self.vertex_centres=self.gridsearch_params[:,0].astype(int)
        self.vertex_centres_dict = [{'vert':k} for k in self.vertex_centres]
        
    def quick_grid_fit(self,sigma_grid):
        
        
        """quick_grid_fit

        Performs fast estimation of vertex centres and sizes using a simple dot product of zscored data.
        Does not complete the regression equation (estimating beta and baseline).


        Parameters
        ----------
        sigma_grid : 1D ndarray
            to be filled in by user


        Returns
        -------
        quick_gridsearch_params: An array containing the gridsearch parameters.
        quick_vertex_centres: An array containing the vertex centres.
        quick_vertex_centres_dict: A dictionary containing the vertex centres.

        """
        
        # Let the model create the timecourses
        self.model.create_grid_predictions(sigma_grid)
        
        self.model.predictions = self.model.predictions.astype('float32')        
        
        # Z-score everything so we can use dot product.
        zdat,zpreds=zscore(self.data.T),zscore(self.model.predictions.T)
        
        # Get all the dot products via np.tensordot.
        fits=np.tensordot(zdat,zpreds,axes=([0],[0]))
        
        # Get the maximum R2 and it's index. 
        max_rsqs,idxs = (np.amax(fits, 1)/zdat.shape[0])**2, np.argmax(fits, axis=1)
        
        self.idxs=idxs
        
        # Output centres, sizes, R2. 
        self.quick_gridsearch_params = np.array([
            self.model.vert_centres_flat[idxs].astype(int),
            self.model.sigmas_flat[idxs],
            max_rsqs]).T
        
        
        # We don't want to submit the vertex_centres for the iterative fitting - these are an additional argument.
        # Save them as .int as bundling them into an array with floats will change their type.
        self.quick_vertex_centres=self.quick_gridsearch_params[:,0].astype(int)
        
        # Bundle this into a dictionary so that we can use this as one of the **args in the iterative fitter
        self.quick_vertex_centres_dict = [{'vert':k} for k in self.quick_vertex_centres]
    
    def get_quick_grid_preds(self,dset='train'):
        
        
        """get_quick_grid_preds

        Returns the best fitting grid predictions from the quick_grid_fit method.


        Parameters
        ----------
        dset : Which dataset to return for (train or test).


        Returns
        -------
        train_predictions.
        OR
        test_predictions.

        """
        
        # Get the predictions of the best grid fits.
        # All we have to do is index the predictions via the index of the best-fitting prediction for each vertex.
        predictions=self.model.predictions[self.idxs,:]
        
        # Assign to object.
        if dset=='train':
            self.train_predictions=predictions
        elif dset=='test':
            self.test_predictions=predictions
    
    def quick_xval(self,test_data,test_stimulus):
        """quick_xval

        Takes the fitted parameters and tests their performance on the out of sample data.


        Parameters
        ----------
        Test data: Data to test predictions on.
        Test stimulus: CFstimulus class associated with test data.

        Returns
        -------
        CV_R2 - the out of sample performance.

        """
        
        fit_stimulus = deepcopy(self.model.stimulus) # Copy the test stimulus.
        self.test_data=test_data # Assign test data
        
        if test_stimulus is not None:    
            # Make the same grid predictions for the test data - therefore assign the new stimulus to the model class.
            self.model.stimulus = test_stimulus
        
        # Now we can generate the same test predictions with the test design matrix.
        self.model.create_grid_predictions(self.model.sigmas,'cart')
        
        # For each vertex, we then take the combination of parameters that provided the best fit to the training data.
        self.get_quick_grid_preds('test')
        
        # We can now put the fit stimulus back. 
        self.model.stimulus = fit_stimulus
        
        # Zscore the data and the preds
        zdat,zpred=zscore(self.test_data,axis=1),zscore(self.test_predictions,axis=1)

        def squaresign(vec):
            """squaresign
                Raises something to a power in a sign-sensive way.
                Useful for if dot products happen to be negative.
            """
            vec2 = (vec**2)*np.sign(vec)
            return vec2
        
        # Get the crossval R2. Here we use np.einsum to calculate the correlations across each row of the test data and the test predictions
        self.xval_R2=squaresign(np.einsum('ij,ij->i',zpred,zdat)/self.test_data.shape[-1])


 
