import numpy as np
from scipy.optimize import fmin_powell, minimize

from joblib import Parallel, delayed


def error_function(
        parameters,
        args,
        data,
        objective_function,
        gradient_objective_function=None):
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
    gradient_objective_function : callable, optional
        This argument is unused but needed because for explicit gradient minimizations,
        the arguments passed to both error function and its gradient are the
        same.
    Returns
    -------
    error : float
        The residual sum of squared errors between the prediction and data.
    """
    error = np.sum((data - objective_function(*list(parameters), **args))**2)/(len(data) * data.var())

    if np.isnan(error):
        return np.inf
    else:
        return error


def gradient_error_function(
        parameters,
        args,
        data,
        objective_function,
        gradient_objective_function):
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
    gradient_objective_function : callable
        Takes `parameters` and `args` and computes the explicit gradient of
        the model objective function.
    Returns
    -------
    gradient : float
        The gradient of the sum of squared errors between the prediction and data.
    """
    gradient = np.sum(-2 * (data - objective_function(*list(parameters), **args))[np.newaxis, ...]
                     * gradient_objective_function(*list(parameters), **args), axis=-1)
    if np.any(np.isnan(gradient)):
        return np.inf
    else:
        return gradient


def iterative_search(gridder, data, start_params, args, xtol=1e-6, ftol=1e-3, verbose=True,
                     bounds=None, gradient_method='numerical', constraints=[], **kwargs):
    """iterative_search

    Generic minimization function called by iterative_fit.
    Do not call this directly. Use iterative_fit instead.

    [description]

    Parameters
    ----------
    gridder : Gridder
        Object that provides the predictions using its
        `return_single_prediction` method
    data : 1D numpy.ndarray
        the data to fit, same dimensions as are returned by
        Gridder's `return_single_prediction` method
    start_params : list or 1D numpy.ndarray
        initial values for the fit
    args : dictionary, arguments to gridder.return_single_prediction that
        are not optimized
    xtol : float, passed to fitting routine
        numerical tolerance on x
    ftol : float, passed to fitting routine
        numerical tolerance on function
    verbose : bool, optional
        whether to have minimizer output.
    bounds : list of tuples, optional
        Bounds for parameter minimization. Must have the same
        length as start_params. The default is None.
    gradient_method : string, optional.
        Can be one of 'numerical' or 'analytic' or None. The default is 'numerical'.

        gradient method for bounded minimization (unbounded minimization does
        not use any gradient). If analytic is selected,
        a suitable 'gradient_single_prediction' must
        be defined in the model gridder.

    **kwargs : TYPE
        DESCRIPTION.

    Raises
    ------
    AssertionError
        Raised if parameters and bounds do not have the same length, or if
        analytic gradient is selected but no gradient function can be found.

    Returns
    -------
    2-tuple
        first element: parameter values,
        second element: rsq value
    """
    if bounds is not None:
        assert len(bounds) == len(
            start_params), "Unequal bounds and parameters"

        if gradient_method == 'analytic':

            assert hasattr(gridder, 'gradient_single_prediction'),\
                "Analytic gradient selected but no gradient function provided"

            if verbose:
                print('Using analytic gradient')

            output = minimize(error_function, start_params, bounds=bounds,
                              args=(
                                  args,
                                  data,
                                  gridder.return_single_prediction,
                                  gridder.gradient_single_prediction),
                              jac=gradient_error_function,
                              method='L-BFGS-B',
                              # default max line searches is 20
                              options=dict(maxls=40, disp=verbose))
        elif gradient_method == 'numerical':
            if verbose:
                print('Using numerical gradient')

            output = minimize(error_function, start_params, bounds=bounds,
                              args=(
                                  args, data, gridder.return_single_prediction),
                              method='L-BFGS-B',
                              # default max line searches is 20
                              options=dict(maxls=40, disp=verbose))
        else:
            if verbose:
                print('Using no-gradient minimization')

            output = minimize(error_function, start_params, bounds=bounds,
                              args=(args, data,
                                    gridder.return_single_prediction),
                              method='trust-constr',
                              constraints=constraints,
                              options=dict(disp=verbose))
                                           #maxiter=3000,
                                           #xtol=1e-10,
                                           #gtol=1e-10))

        return np.r_[output['x'], 1 -
                     (output['fun'])]

    else:

        output = fmin_powell(
            error_function,
            start_params,
            xtol=xtol,
            ftol=ftol,
            args=(
                args,
                data,
                gridder.return_single_prediction),
            full_output=True,
            disp=verbose)

        return np.r_[output[0], 1 - (output[1])]


class Fitter:
    """Fitter

    Superclass for classes that implement the different fitting methods,
    for a given model. It contains 2D-data and leverages a Gridder object.

    Data should be two-dimensional so that all bookkeeping with regard to voxels,
    electrodes, etc is done by the user. Generally, a Fitter class should implement
    both a `grid_fit` and an `interative_fit` method to be run in sequence.

    """

    def __init__(self, data, gridder, n_jobs=1, **kwargs):
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

        self.__dict__.update(kwargs)

        self.n_units = self.data.shape[0]
        self.n_timepoints = self.data.shape[-1]

        self.data_var = self.data.var(axis=-1)

    def iterative_fit(self,

                      rsq_threshold,
                      verbose=False,
                      starting_params=None,
                      bounds=None,
                      gradient_method='numerical',
                      fit_hrf=False,
                      args={},
                      constraints=[]):
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
        starting_params : ndarray of size [units, model params +1], optional
            Explicit start for iterative fit. The default is None.
        bounds : list of tuples, optional
            Bounds for parameter minimization. The default is None.
        gradient_method : string, optional.
            Can be one of 'numerical' or 'analytic' or None. The default is 'numerical'.
            If analytic, a gradient_single_prediction function must be found
            in the model Gridder class.
        fit_hrf : boolean, optional
            Whether or not to fit two extra parameters for hrf derivative and
            dispersion. The default is False.
        args : dictionary, optional
            Further arguments passed to iterative_search. The default is {}.

        Returns
        -------
        None.

        """

        self.bounds = bounds
        self.gradient_method = gradient_method.lower()
        self.fit_hrf = fit_hrf

        if starting_params is None:
            assert hasattr(
                self, 'gridsearch_params'), 'First use self.grid_fit,\
            or provide explicit starting parameters!'

            self.starting_params = self.gridsearch_params

            if self.fit_hrf:
                self.starting_params = np.insert(
                    self.starting_params, -1, 1.0, axis=-1)
                self.starting_params = np.insert(
                    self.starting_params, -1, 0.0, axis=-1)

        else:
            self.starting_params = starting_params

        self.rsq_mask = self.starting_params[:, -1] > rsq_threshold

        self.iterative_search_params = np.zeros_like(self.starting_params)

        if self.rsq_mask.sum()>0:
            iterative_search_params = Parallel(self.n_jobs, verbose=verbose)(
                delayed(iterative_search)(self.gridder,
                                          data,
                                          start_params,
                                          args=args,
                                          verbose=verbose,
                                          bounds=self.bounds,
                                          gradient_method=self.gradient_method,
                                          constraints=constraints)
                for (data, start_params) in zip(self.data[self.rsq_mask], self.starting_params[self.rsq_mask][:, :-1]))
            self.iterative_search_params[self.rsq_mask] = np.array(
                iterative_search_params)


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
                 n_batches=1000,
                 pos_prfs_only=True):
        """grid_fit

        performs grid fit using provided grids and predictor definitions


        Parameters
        ----------
        ecc_grid : 1D ndarray
            to be filled in by user
        polar_grid : 1D ndarray
            to be filled in by user
        size_grid : 1D ndarray
            to be filled in by user
        verbose : boolean, optional
            print output. The default is False.
        n_batches : int, optional
            The grid fit is performed in parallel over n_batches of units.
            Batch parallelization is faster than single-unit
            parallelization and of sequential computing.

        Returns
        -------
        None.

        """
        # let the gridder create the timecourses
        self.gridder.create_grid_predictions(ecc_grid=ecc_grid,
                                             polar_grid=polar_grid,
                                             size_grid=size_grid)
        self.gridder.predictions = self.gridder.predictions.astype('float32')

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
                if pos_prfs_only:
                    if np.any(slopes>0):
                        resid[slopes<=0] = +np.inf

                best_pred_voxel = np.argmin(resid)

                rsq = 1 - resid[best_pred_voxel]**2 / \
                    (n_timepoints * data_var[num])

                result[idx, :] = best_pred_voxel, rsq, baselines[best_pred_voxel], slopes[best_pred_voxel]

            return result

        # bookkeeping
        sum_preds = np.sum(self.gridder.predictions, axis=-1)
        square_norm_preds = np.linalg.norm(
            self.gridder.predictions, axis=-1, ord=2)**2

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
                predictions=self.gridder.predictions,
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
            self.gridder.xs.ravel()[max_rsqs],
            self.gridder.ys.ravel()[max_rsqs],
            self.gridder.sizes.ravel()[max_rsqs],
            self.best_fitting_beta,
            self.best_fitting_baseline,
            self.gridsearch_r2
        ]).T


class Extend_Iso2DGaussianFitter(Iso2DGaussianFitter):
    """

    Generic superclass to extend the Gaussian Fitter. If an existing
    Iso2DGaussianFitter object with iterative_search_params is provided, the
    prf position, size, and rsq parameters will be used for further minimizations.

    """

    def __init__(self, data, gridder, n_jobs=1,
                 previous_gaussian_fitter=None,
                 **kwargs):
        """

        Parameters
        ----------
        data : numpy.ndarray, 2D
            input data. First dimension units, Second dimension time
        gridder : prfpy.Gridder
            Gridder object that provides the grid and iterative search
            predictions.
        n_jobs : int, optional
            number of jobs to use in parallelization (iterative search), by default 1
        previous_gaussian_fitter : Iso2DGaussianFitter, optional
            Must have iterative_search_params. The default is None.
        **kwargs : TYPE
            DESCRIPTION.

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

        super().__init__(data, gridder, n_jobs=n_jobs, **kwargs)

    def insert_new_model_params(self, old_params):
        """
        Function to insert new model parameters starting values for iterfitting.
        To be redefined appropriately for each model (see below for examples).
        If `grid_fit` is defined and performed, `self.gridsearch_params` take
        precedence, and this function becomes unnecessary.

        Parameters
        ----------
        old_params : ndarray [n_units, 6]
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
                      gradient_method='numerical',
                      fit_hrf=False,
                      args={},
                      constraints=[]):
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
        starting_params : ndarray of size [units, model_params +1], optional
            Explicit start for minimization. The default is None.
        bounds : list of tuples, optional
            Bounds for parameter minimization. The default is None.
        gradient_method : string, optional.
            Can be one of 'numerical' or 'analytic' or None. The default is 'numerical'.
        fit_hrf : boolean, optional
            Whether or not to fit 2 extra parameters for hrf derivative and
            dispersion. The default is False.
        args : dictionary, optional
            Further arguments passed to iterative_search. The default is {}.

        Returns
        -------
        None.

        """

        if starting_params is None and not hasattr(
            self, 'gridsearch_params') and hasattr(
                self, 'previous_gaussian_fitter'):

            starting_params = self.insert_new_model_params(
                self.previous_gaussian_fitter.iterative_search_params)

            # handling fit_hrf discrepancy between previous and current fitter
            # for now, i am enforcing "consistency" with previous gaussian fit:
            # if hrf was fit, it will be fit for new model too, and vice
            # versa. Could easily be changed for more flexibility.
            if self.previous_gaussian_fitter.fit_hrf != fit_hrf:

                print("Warning: fit_hrf was " + str(
                    self.previous_gaussian_fitter.fit_hrf) + " in previous_\
                      gaussian_fit. Overriding current fit_hrf to avoid inconsistency.")

                fit_hrf = self.previous_gaussian_fitter.fit_hrf

        super().iterative_fit(rsq_threshold=rsq_threshold,
                              verbose=verbose,
                              starting_params=starting_params,
                              bounds=bounds,
                              gradient_method=gradient_method,
                              fit_hrf=fit_hrf,
                              args=args,
                              constraints=constraints)


class CSS_Iso2DGaussianFitter(Extend_Iso2DGaussianFitter):
    """CSS_Iso2DGaussianFitter

    Compressive Spatial Summation model
    """

    def insert_new_model_params(self, old_params):
        """
        Parameters
        ----------
        old_params : ndarray [n_units, 6]
            Previous Gaussian fitter parameters and rsq.

        Returns
        -------
        new_params : ndarray [n_units, 7]
            Starting parameters and rsq for CSS iterative fit.

        """
        # insert CSS exponent
        new_params = np.insert(old_params, 5, 1.0, axis=-1)
        return new_params


class DoG_Iso2DGaussianFitter(Extend_Iso2DGaussianFitter):
    """DoG_Iso2DGaussianFitter

    Difference of Gaussians model
    """

    def insert_new_model_params(self, old_params):
        """
        Parameters
        ----------
        old_params : ndarray [n_units, 6]
            Previous Gaussian fitter parameters and rsq.

        Returns
        -------
        new_params : ndarray [n_units, 8]
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
        old_params : ndarray [n_units, 6]
            Previous Gaussian fitter parameters and rsq.

        Returns
        -------
        new_params : ndarray [n_units, 10]
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
                 n_batches=1000,
                 rsq_threshold=0.1,
                 pos_prfs_only=True):
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
            Array of surround size values.
        neural_baseline_grid : 1D ndarray
            Array of neural baseline values.
        surround_baseline_grid : 1D ndarray
            Array of surround baseline values.
        gaussian_params : ndarray [n_units, 4], optional
            The Gaussian parms [x position, y position, prf size, rsq] can be
            provided explicitly. If not, a previous_gaussian_fitter must be
            provided. The default is None.
        verbose : boolean, optional
            print output. The default is False.
        n_batches : int, optional
            Number of voxel batches. The default is 1000.
        rsq_threshold : float, optional
            rsq threshold for grid fitting. The default is 0.1.

        Raises
        ------
        ValueError
            Raised if there is no previous_gaussian_fitter or gaussian params.

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

        if gaussian_params is not None and gaussian_params.shape == (
                self.n_units, 4):
            self.gaussian_params = gaussian_params.astype('float32')
        elif hasattr(self, 'previous_gaussian_fitter'):
            starting_params_grid = self.previous_gaussian_fitter.iterative_search_params
            self.gaussian_params = np.concatenate(
                (starting_params_grid[:, :3], starting_params_grid[:, -1][..., np.newaxis]), axis=-1)
        else:
            print('Please provide suitable [n_units, 4] gaussian_params,\
                  or previous_gaussian_fitter')
            raise ValueError

        self.gridsearch_rsq_mask = self.gaussian_params[:, -1] > rsq_threshold

        # this function analytically computes best-fit rsq, slope, and baseline
        # for a given batch of units (faster than scipy/numpy lstsq).
        def rsq_betas_for_batch(data,
                                vox_nums,
                                n_predictions,
                                n_timepoints,
                                data_var,
                                sa, ss, nb, sb,
                                gaussian_params):

            result = np.zeros((data.shape[0], 4), dtype='float32')

            for vox_data, vox_num, idx in zip(
                data, vox_nums, np.arange(
                    data.shape[0])):

                # let the gridder create the timecourses, per voxel, since the
                # gridding is over new parameters, while size and position
                # are obtained from previous Gaussian fit
                predictions = self.gridder.create_grid_predictions(
                    gaussian_params[vox_num, :-1], n_predictions, n_timepoints, sa, ss, nb, sb)
                # bookkeeping
                sum_preds = np.sum(predictions, axis=-1)
                square_norm_preds = np.linalg.norm(
                    predictions, axis=-1, ord=2)**2
                sumd = np.sum(vox_data)

                # best possible slopes and baselines
                slopes = (n_timepoints * np.dot(vox_data, predictions.T) - sumd *
                          sum_preds) / (n_timepoints * square_norm_preds - sum_preds**2)
                baselines = (sumd - slopes * sum_preds) / n_timepoints

                # find best prediction and store relevant data
                resid = np.linalg.norm((vox_data -
                                        slopes[..., np.newaxis] *
                                        predictions -
                                        baselines[..., np.newaxis]), ord=2, axis=-
                                       1)

                #to enforce, if possible, positive prf amplitude & neural baseline
                if pos_prfs_only:
                    if np.any(slopes>0):
                        resid[slopes<=0] = +np.inf

                best_pred_voxel = np.argmin(resid)

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
            self.gaussian_params[self.gridsearch_rsq_mask, 0],
            self.gaussian_params[self.gridsearch_rsq_mask, 1],
            self.gaussian_params[self.gridsearch_rsq_mask, 2],
            self.best_fitting_beta,
            self.best_fitting_baseline,
            self.sa[max_rsqs],
            self.ss[max_rsqs],
            self.nb[max_rsqs] * self.best_fitting_beta,
            self.sb[max_rsqs],
            self.gridsearch_r2
        ]).T
