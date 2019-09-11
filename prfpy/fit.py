import numpy as np
from scipy.optimize import fmin_powell, minimize
import bottleneck as bn

from joblib import Parallel, delayed


def error_function(parameters, args, data, objective_function):
    """error_function

    Generic error function.

    [description]

    Parameters
    ----------
    parameters : tuple
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
    return bn.nansum((data - objective_function(*list(parameters), **args))**2)


def gradient_error_function(parameters, args, data, objective_function, gradient_objective_function):
    """error_function

    Generic gradieht error function.

    [description]

    Parameters
    ----------
    parameters : tuple
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
    return bn.nansum(-2*(data - objective_function(*list(parameters), **args))[np.newaxis, ...]
                     * gradient_objective_function(*list(parameters), **args), axis=-1)


def iterative_search(gridder, data, start_params, args, verbose=True, **kwargs):
    """iterative_search

    function to be called using joblib's Parallel function for the iterative search stage.

    [description]

    Parameters
    ----------
    gridder : Gridder
        Object that provides the predictions using its
        `return_single_prediction` method
    data : 1D numpy.ndarray
        the data to fit, same dimensions as are returned by
        Gridder's `return_single_prediction` method
    grid_params : tuple [float]
        initial values for the fit
    args : dictionary, arguments to gridder.return_single_prediction that
        are not optimized
    verbose : bool, optional
        whether to have fminpowell puke everything out.

    Returns
    -------
    2-tuple
        first element: parameter values,
        second element: rsq value
    """
    if kwargs['bounds'] is not None:

        if kwargs['gradient_method'] == 'analytic':

            if not hasattr(gridder, 'gradient_single_prediction'):
                print('Analytic gradient selected but no gradient function provided')
                raise IOError

            if verbose:
                print('Using analytic gradient')

            output = minimize(error_function, start_params, bounds=kwargs['bounds'],
                              args=(args, data, gridder.return_single_prediction),
                              jac=gradient_error_function,
                              method='L-BFGS-B',
                              options=dict(disp=verbose, maxls=300, ftol=1e-80))
        elif kwargs['gradient_method'] == 'numerical':
            if verbose:
                print('Using numerical gradientu')

            output = minimize(error_function, start_params, bounds=kwargs['bounds'],
                              args=(args, data, gridder.return_single_prediction),
                              method='L-BFGS-B',
                              options=dict(disp=verbose, maxls=300, ftol=1e-80))
        else:
            if verbose:
                print('Using no-gradient minimization')
            output = minimize(error_function, start_params, bounds=kwargs['bounds'],
                              args=(args, data,
                                    gridder.return_single_prediction),
                              method='trust-constr',
                              options=dict(disp=verbose))

        return np.r_[output['x'], 1 - (output['fun'] / (len(data) * data.var()))]

    else:

        output = fmin_powell(error_function, start_params, xtol=1e-6, ftol=1e-6,
                             args=(args, data, gridder.return_single_prediction),
                             full_output=True, disp=verbose)

        return np.r_[output[0], 1 - (output[1] / (len(data) * data.var()))]


class Fitter:
    """Fitter

    Superclass for classes that implement the different fitting methods,
    for a given model. It contains 2D-data and leverages a Gridder object.

    Data should be two-dimensional so that all bookkeeping with regard to voxels,
    electrodes, etc is done by the user. Generally, a Fitter class should implement
    both a `grid_fit` and an `interative_fit` method to be run in sequence.

    """

    def __init__(self, data, gridder, n_jobs=1, bounds=None,
                 gradient_method='numerical',
                 fit_hrf=False,
                 **kwargs):
        """__init__ sets up data and gridder

        Parameters
        ----------
        data : numpy.ndarray, 2D
            input data. First dimension units, Second dimension time
        gridder : prfpy.Gridder
            Gridder object that provides the grid and iterative search
            predictions.
        n_jobs : int, optional
            number of jobs to use in parallelization (iterative search), by default 1
        """
        assert len(data.shape) == 2, \
            "input data should be two-dimensional, with first dimension units and second dimension time"
        self.data = data.astype('float32')
        self.gridder = gridder
        self.n_jobs = n_jobs
        self.bounds = bounds
        self.gradient_method = gradient_method
        self.fit_hrf = fit_hrf
        self.__dict__.update(kwargs)

        self.n_units = self.data.shape[0]
        self.n_timepoints = self.data.shape[-1]

        # immediately convert nans to nums
        self.data = np.nan_to_num(data)
        self.data_var = self.data.var(axis=-1)


class Iso2DGaussianFitter(Fitter):
    """Iso2DGaussianFitter

    Class that implements the different fitting methods
    on a two-dimensional isotropic Gaussian pRF model,
    leveraging a Gridder object.

    """

    def grid_fit(self,
                 ecc_grid,
                 polar_grid,
                 size_grid,
                 verbose=False,
                 n_batches=1000):
        """grid_fit

        performs grid fit using provided grids and predictor definitions

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
        # let the gridder create the timecourses
        self.gridder.create_grid_predictions(ecc_grid=ecc_grid,
                                             polar_grid=polar_grid,
                                             size_grid=size_grid)

        def rsq_betas_for_pred_analytic(data, vox_num, predictions, n_timepoints, data_var, sum_preds, square_norm_preds):
            result = np.zeros((data.shape[0], 4), dtype='float32')
            for vox_data, num, idx in zip(data, vox_num, np.arange(data.shape[0])):
                sumd = np.sum(vox_data)

                slopes = (n_timepoints*np.dot(vox_data, predictions.T)-sumd*sum_preds) /\
                    (n_timepoints*square_norm_preds-sum_preds**2)
                baselines = (sumd - slopes*sum_preds)/n_timepoints

                resid = np.linalg.norm(
                    (vox_data-slopes[..., np.newaxis]*predictions-baselines[..., np.newaxis]), axis=-1, ord=2)

                best_pred_voxel = np.argmin(resid)

                rsq = 1-resid[best_pred_voxel]**2/(n_timepoints*data_var[num])

                result[idx, :] = best_pred_voxel, rsq, baselines[best_pred_voxel], slopes[best_pred_voxel]

            return result

        self.gridder.predictions = self.gridder.predictions.astype('float32')
        self.sum_preds = np.sum(self.gridder.predictions, axis=-1)
        self.square_norm_preds = np.linalg.norm(
            self.gridder.predictions, axis=-1, ord=2)**2

        split_indices = np.array_split(
            np.arange(self.data.shape[0]), n_batches)
        data_batches = np.array_split(self.data, n_batches, axis=0)
        if verbose:
            print("Each batch contains " +
                  str(data_batches[0].shape[0])+" voxels.")

        grid_search_rbs = Parallel(self.n_jobs, verbose=11)(
            delayed(rsq_betas_for_pred_analytic)(
                data=data,
                vox_num=vox_num,
                predictions=self.gridder.predictions,
                n_timepoints=self.n_timepoints,
                data_var=self.data_var,
                sum_preds=self.sum_preds,
                square_norm_preds=self.square_norm_preds)
            for data, vox_num in zip(data_batches, split_indices))

        grid_search_rbs = np.concatenate(grid_search_rbs, axis=0)

        max_rsqs = grid_search_rbs[:, 0].astype('int')
        self.gridsearch_r2 = grid_search_rbs[:, 1]
        self.best_fitting_baseline = grid_search_rbs[:, 2]
        self.best_fitting_beta = grid_search_rbs[:, 3]

        self.gridsearch_params = np.array([
            self.gridder.xs.ravel()[max_rsqs],
            self.gridder.ys.ravel()[max_rsqs],
            self.gridder.sizes.ravel()[max_rsqs],
            self.best_fitting_beta,
            self.best_fitting_baseline,
            self.gridsearch_r2
        ]).T

    def iterative_fit(self,
                      rsq_threshold,
                      verbose=False,
                      starting_params=None,
                      args={}):
        if starting_params is None:
            if hasattr(self, 'gridsearch_params'):
                self.starting_params = self.gridsearch_params
            else:
                print('First use self.grid_fit, or provide starting parameters!')
                raise IOError
        else:
            self.starting_params = starting_params

        self.rsq_mask = self.starting_params[:, -1] > rsq_threshold


        if self.fit_hrf:
            self.starting_params = np.insert(
                self.starting_params, -1, 1.0, axis=-1)
            self.starting_params = np.insert(
                self.starting_params, -1, 0.0, axis=-1)


        # create output array
        self.iterative_search_params = np.zeros_like(self.starting_params)

        iterative_search_params = Parallel(self.n_jobs, verbose=verbose)(
            delayed(iterative_search)(self.gridder,
                                      data,
                                      start_params,
                                      args=args,
                                      verbose=verbose,
                                      bounds=self.bounds,
                                      gradient_method=self.gradient_method)
            for (data, start_params) in zip(self.data[self.rsq_mask], self.starting_params[self.rsq_mask][:, :-1]))
        self.iterative_search_params[self.rsq_mask] = np.array(
            iterative_search_params)

class Extend_Iso2DGaussianFitter(Iso2DGaussianFitter):


    def __init__(self, data, gridder, n_jobs=1, bounds=None,
                 gradient_method='numerical',
                 fit_hrf=False,
                 previous_gaussian_fitter=None,
                 **kwargs):

        if previous_gaussian_fitter is not None:
            self.previous_gaussian_fitter = previous_gaussian_fitter

        super().__init__(data, gridder, n_jobs=1, bounds=None,
                 gradient_method='numerical',
                 fit_hrf=False
                 **kwargs)

class CSS_Iso2DGaussianFitter(Extend_Iso2DGaussianFitter):

    def grid_fit(self,
                 ecc_grid,
                 polar_grid,
                 size_grid,
                 n_grid,
                 verbose=False,
                 n_batches=1000):
        """grid_fit

        performs grid fit using provided grids and predictor definitions

        [description]

        Parameters
        ----------
        ecc_grid : list
            to be filled in by user
        polar_grid : list
            to be filled in by user
        size_grid : list
            to be filled in by user
        n_grid : list, 
        """
        # let the gridder create the timecourses
        self.gridder.create_grid_predictions(ecc_grid=ecc_grid,
                                             polar_grid=polar_grid,
                                             size_grid=size_grid,
                                             n_grid=n_grid)

        def rsq_betas_for_pred_analytic(data, vox_num, predictions, n_timepoints, data_var, sum_preds, square_norm_preds):
            result = np.zeros((data.shape[0], 4), dtype='float32')
            for vox_data, num, idx in zip(data, vox_num, np.arange(data.shape[0])):
                sumd = np.sum(vox_data)

                slopes = (n_timepoints*np.dot(vox_data, predictions.T)-sumd*sum_preds) /\
                    (n_timepoints*square_norm_preds-sum_preds**2)
                baselines = (sumd - slopes*sum_preds)/n_timepoints

                resid = np.linalg.norm(
                    (vox_data-slopes[..., np.newaxis]*predictions-baselines[..., np.newaxis]), axis=-1, ord=2)

                best_pred_voxel = np.argmin(resid)

                rsq = 1-resid[best_pred_voxel]**2/(n_timepoints*data_var[num])

                result[idx, :] = best_pred_voxel, rsq, baselines[best_pred_voxel], slopes[best_pred_voxel]

            return result

        self.gridder.predictions = self.gridder.predictions.astype('float32')
        self.sum_preds = np.sum(self.gridder.predictions, axis=-1)
        self.square_norm_preds = np.linalg.norm(
            self.gridder.predictions, axis=-1, ord=2)**2

        split_indices = np.array_split(
            np.arange(self.data.shape[0]), n_batches)
        data_batches = np.array_split(self.data, n_batches, axis=0)
        if verbose:
            print("Each batch contains " +
                  str(data_batches[0].shape[0])+" voxels.")

        grid_search_rbs = Parallel(self.n_jobs, verbose=11)(
            delayed(rsq_betas_for_pred_analytic)(
                data=data,
                vox_num=vox_num,
                predictions=self.gridder.predictions,
                n_timepoints=self.n_timepoints,
                data_var=self.data_var,
                sum_preds=self.sum_preds,
                square_norm_preds=self.square_norm_preds)
            for data, vox_num in zip(data_batches, split_indices))

        grid_search_rbs = np.concatenate(grid_search_rbs, axis=0)

        max_rsqs = grid_search_rbs[:, 0].astype('int')
        self.gridsearch_r2 = grid_search_rbs[:, 1]
        self.best_fitting_baseline = grid_search_rbs[:, 2]
        self.best_fitting_beta = grid_search_rbs[:, 3]

        self.gridsearch_params = np.array([
            self.gridder.xs.ravel()[max_rsqs],
            self.gridder.ys.ravel()[max_rsqs],
            self.gridder.sizes.ravel()[max_rsqs],
            self.best_fitting_beta,
            self.best_fitting_baseline,
            self.gridder.ns.ravel()[max_rsqs],
            self.gridsearch_r2
        ]).T

    def iterative_fit(self,
                      rsq_threshold,
                      verbose=False,
                      starting_params=None,
                      args={}):
        if starting_params is None:
            if hasattr(self, 'gridsearch_params'):
                if self.fit_hrf:
                    #insert hrf params
                    self.starting_params =  np.insert(self.gridsearch_params, -1, 1.0, axis=-1)
                    self.starting_params =  np.insert(self.starting_params, -1, 0.0, axis=-1)
                else:
                    self.starting_params = self.gridsearch_params

            elif hasattr(self, 'previous_gaussian_fitter'):
                #insert CSS exponent
                self.starting_params = np.insert(self.previous_gaussian_fitter.iterative_search_params, 5, 1.0, axis=-1)

            else:
                print('First use self.grid_fit, Iso2DGaussianFitter.iterative_fit,\
                      or provide explicit starting parameters!')
                raise IOError
        else:
            self.starting_params = starting_params

        self.rsq_mask = self.starting_params[:, -1] > rsq_threshold


        # create output array
        self.iterative_search_params = np.zeros_like(self.starting_params)

        iterative_search_params = Parallel(self.n_jobs, verbose=verbose)(
            delayed(iterative_search)(self.gridder,
                                      data,
                                      start_params,
                                      args=args,
                                      verbose=verbose,
                                      bounds=self.bounds,
                                      gradient_method=self.gradient_method)
            for (data, start_params) in zip(self.data[self.rsq_mask], self.starting_params[self.rsq_mask][:, :-1]))
        self.iterative_search_params[self.rsq_mask] = np.array(
            iterative_search_params)

class Norm_Iso2DGaussianFitter(Extend_Iso2DGaussianFitter):
    """Norm_Iso2DGaussianFitter

    Class that implements a grid fit on a two-dimensional isotropic
    Gaussian pRF model, leveraging a Gridder object.
    The gridder result is used as starting guess to fit the Normalization model
    with an iterative fitting procedure

    """

    def grid_fit(self,
                 surround_amplitude_grid,
                 surround_size_grid,
                 neural_baseline_grid,
                 surround_baseline_grid,
                 gaussian_params=None,
                 verbose=False,
                 n_batches=1000,
                 rsq_threshold=0.1):
        """grid_fit

        performs grid fit using provided grids and predictor definitions

        [description]

        Parameters
        ----------
        gaussian_params: [units, 4] array, contining x-position, y-position,
        size of the gaussian PRF, and RSQ, previously obtained.

        """
        # setting up grid for norm model new params
        self.sa, self.ss, self.nb, self.sb = np.meshgrid(
            surround_amplitude_grid, surround_size_grid,
            neural_baseline_grid, surround_baseline_grid)


        self.sa = self.sa.ravel()
        self.ss = self.ss.ravel()
        self.nb = self.nb.ravel()
        self.sb = self.sb.ravel()

        self.n_predictions = len(self.nb)

        if gaussian_params is not None and gaussian_params.shape==(self.n_units,4):
            self.gaussian_params = gaussian_params.astype('float32')
        elif hasattr(self, 'previous_gaussian_fitter'):
            starting_params_grid = self.previous_gaussian_fitter.iterative_search_params
            self.gaussian_params = np.concatenate((starting_params_grid[:,:3], starting_params_grid[:,-1][...,np.newaxis]), axis=-1)
        else:
            print('Please provide suitable [n_units, 4] gaussian params,\
                  or previous gaussian fitter')
            raise IOError

        self.gridsearch_rsq_mask = self.gaussian_params[:, -1] > rsq_threshold

        def rsq_betas_for_batch(data,
                                vox_nums,
                                n_predictions,
                                n_timepoints,
                                data_var,
                                sa, ss, nb, sb,
                                gaussian_params):

            result = np.zeros((data.shape[0], 4), dtype='float32')

            for vox_data, vox_num, idx in zip(data, vox_nums, np.arange(data.shape[0])):

                # let the gridder create the timecourses
                predictions = self.gridder.create_grid_predictions(gaussian_params[vox_num, :-1],
                                                                   n_predictions,
                                                                   n_timepoints,
                                                                   sa, ss, nb, sb)
                # bookkeeping
                sum_preds = np.sum(predictions, axis=-1)
                square_norm_preds = np.linalg.norm(
                    predictions, axis=-1, ord=2)**2
                sumd = np.sum(vox_data)

                # best possible slopes and baselines
                slopes = (n_timepoints*np.dot(vox_data, predictions.T)-sumd*sum_preds) /\
                    (n_timepoints*square_norm_preds-sum_preds**2)
                baselines = (sumd - slopes*sum_preds)/n_timepoints

                # find best prediction and store relevant data
                resid = np.linalg.norm(
                    (vox_data-slopes[..., np.newaxis]*predictions-baselines[..., np.newaxis]), ord=2, axis=-1)

                best_pred_voxel = np.argmin(resid)

                rsq = 1-resid[best_pred_voxel]**2 / \
                    (n_timepoints*data_var[vox_num])

                result[idx, :] = best_pred_voxel, rsq, baselines[best_pred_voxel], slopes[best_pred_voxel]

            return result

        # masking and splitting data
        split_indices = np.array_split(np.arange(self.data.shape[0])[
                                       self.gridsearch_rsq_mask], n_batches)
        data_batches = np.array_split(
            self.data[self.gridsearch_rsq_mask], n_batches, axis=0)

        if verbose:
            print("Each batch contains approx. " +
                  str(data_batches[0].shape[0])+" voxels.")

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
                gaussian_params=self.gaussian_params)
            for data, vox_nums in zip(data_batches, split_indices))

        grid_search_rbs = np.concatenate(grid_search_rbs, axis=0)

        # store results
        max_rsqs = grid_search_rbs[:, 0].astype('int')
        self.gridsearch_r2 = grid_search_rbs[:, 1]
        self.best_fitting_baseline = grid_search_rbs[:, 2]
        self.best_fitting_beta = grid_search_rbs[:, 3]

        self.gridsearch_params = np.zeros((self.n_units, 10))

        self.gridsearch_params[self.gridsearch_rsq_mask] = np.array([
            gaussian_params[self.gridsearch_rsq_mask, 0],
            gaussian_params[self.gridsearch_rsq_mask, 1],
            gaussian_params[self.gridsearch_rsq_mask, 2],
            self.best_fitting_beta,
            self.best_fitting_baseline,
            self.sa[max_rsqs],
            self.ss[max_rsqs],
            self.nb[max_rsqs]*self.best_fitting_beta,
            self.sb[max_rsqs],
            self.gridsearch_r2
        ]).T

    def iterative_fit(self,
                      rsq_threshold,
                      verbose=False,
                      starting_params=None,
                      args={}):
        if starting_params is None:
            if hasattr(self, 'gridsearch_params'):
                if self.fit_hrf:
                    #insert hrf params
                    self.starting_params =  np.insert(self.gridsearch_params, -1, 1.0, axis=-1)
                    self.starting_params =  np.insert(self.starting_params, -1, 0.0, axis=-1)
                else:
                    self.starting_params = self.gridsearch_params

            elif hasattr(self, 'previous_gaussian_fitter'):
                print("Warning: iterfitting normalization model without grid stage")
                #surround amplitude
                self.starting_params = np.insert(self.previous_gaussian_fitter.iterative_search_params, 5, 0.0, axis=-1)
                #surround size
                self.starting_params = np.insert(self.starting_params, 6, self.gridder.stimulus.max_ecc, axis=-1)
                #neural baseline
                self.starting_params = np.insert(self.starting_params, 7, 0.0, axis=-1)
                #surround baseline
                self.starting_params = np.insert(self.starting_params, 8, 1.0, axis=-1)

            else:
                print('First use self.grid_fit, Iso2DGaussianFitter.iterative_fit,\
                      or provide explicit starting parameters!')
                raise IOError
        else:
            self.starting_params = starting_params

        self.rsq_mask = self.starting_params[:, -1] > rsq_threshold


        # create output array
        self.iterative_search_params = np.zeros_like(self.starting_params)

        iterative_search_params = Parallel(self.n_jobs, verbose=verbose)(
            delayed(iterative_search)(self.gridder,
                                      data,
                                      start_params,
                                      args=args,
                                      verbose=verbose,
                                      bounds=self.bounds,
                                      gradient_method=self.gradient_method)
            for (data, start_params) in zip(self.data[self.rsq_mask], self.starting_params[self.rsq_mask][:, :-1]))
        self.iterative_search_params[self.rsq_mask] = np.array(
            iterative_search_params)


class DoG_Iso2DGaussianFitter(Extend_Iso2DGaussianFitter):
    """DoG_Iso2DGaussianFitter

    Class that implements a grid fit on a two-dimensional isotropic
    difference of Gaussians pRF model, leveraging a Gridder object.
    The gridder result is used as starting guess to fit the DoG model
    with an iterative fitting procedure

    """
    def iterative_fit(self,
                      rsq_threshold,
                      verbose=False,
                      starting_params=None,
                      args={}):
        if starting_params is None:
            if hasattr(self, 'gridsearch_params'):
                if self.fit_hrf:
                    #insert hrf params
                    self.starting_params =  np.insert(self.gridsearch_params, -1, 1.0, axis=-1)
                    self.starting_params =  np.insert(self.starting_params, -1, 0.0, axis=-1)
                else:
                    self.starting_params = self.gridsearch_params

            elif hasattr(self, 'previous_gaussian_fitter'):
                #surround amplitude
                self.starting_params = np.insert(self.previous_gaussian_fitter.iterative_search_params, 5, 0.0, axis=-1)
                #surround size
                self.starting_params = np.insert(self.starting_params, 6, self.gridder.stimulus.max_ecc, axis=-1)

            else:
                print('First use self.grid_fit, Iso2DGaussianFitter.iterative_fit,\
                      or provide explicit starting parameters!')
                raise IOError
        else:
            self.starting_params = starting_params

        self.rsq_mask = self.starting_params[:, -1] > rsq_threshold


        # create output array
        self.iterative_search_params = np.zeros_like(self.starting_params)

        iterative_search_params = Parallel(self.n_jobs, verbose=verbose)(
            delayed(iterative_search)(self.gridder,
                                      data,
                                      start_params,
                                      args=args,
                                      verbose=verbose,
                                      bounds=self.bounds,
                                      gradient_method=self.gradient_method)
            for (data, start_params) in zip(self.data[self.rsq_mask], self.starting_params[self.rsq_mask][:, :-1]))
        self.iterative_search_params[self.rsq_mask] = np.array(
            iterative_search_params)
