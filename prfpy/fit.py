import numpy as np
import scipy as sp
from scipy.optimize import fmin_powell, minimize
import bottleneck as bn
from tqdm import tqdm

from joblib import Parallel, delayed


def error_function(parameters, args, data, objective_function, gradient_objective_function=None):
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
    return bn.nansum(-2*(data - objective_function(*list(parameters), **args))[np.newaxis, ...]
                     * gradient_objective_function(*list(parameters), **args), axis=-1)


def iterative_search(gridder, data, grid_params, args, verbose=True, **kwargs):
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
                print('Using analytic gradientUa')

            output = minimize(error_function, grid_params, bounds=kwargs['bounds'],
                              args=(args, data, gridder.return_single_prediction),
                              jac=gradient_error_function,
                              method='L-BFGS-B',
                              options=dict(disp=verbose, maxls=300, ftol=1e-80))
        elif kwargs['gradient_method'] == 'numerical':
            if verbose:
                print('Using numerical gradient')

            output = minimize(error_function, grid_params, bounds=kwargs['bounds'],
                              args=(args, data,
                                    gridder.return_single_prediction),
                              method='L-BFGS-B',
                              options=dict(disp=verbose, maxls=300, ftol=1e-80))
        else:
            if verbose:
                print('Using no-gradient minimization')
            output = minimize(error_function, grid_params, bounds=kwargs['bounds'],
                              args=(args, data,
                                    gridder.return_single_prediction),
                              method='trust-constr',
                              options=dict(disp=verbose))

        return np.r_[output['x'], 1 - (output['fun'] / (len(data) * data.var()))]

    else:

        output = fmin_powell(error_function, grid_params, xtol=1e-6, ftol=1e-6,
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
        self.data = data
        self.gridder = gridder
        self.n_jobs = n_jobs
        self.__dict__.update(kwargs)

        self.n_units = self.data.shape[0]
        self.n_timepoints = self.data.shape[-1]

        # immediately convert nans to nums
        self.data = np.nan_to_num(data)
        self.data_var = self.data.var(axis=-1)

        if 'bounds' in self.__dict__:
            self.bounds = self.__dict__['bounds']
        else:
            self.bounds = None

        if 'gradient_method' in self.__dict__:
            self.gradient_method = self.__dict__['gradient_method']
        else:
            self.gradient_method = 'numerical'


class Iso2DGaussianFitter(Fitter):
    """Iso2DGaussianFitter

    Class that implements the different fitting methods
    on a two-dimensional isotropic Gaussian pRF model,
    leveraging a Gridder object.

    """

    def __init__(self, data, gridder, n_jobs=1, fit_css=False, **kwargs):
        self.fit_css = fit_css
        super().__init__(data=data, gridder=gridder, n_jobs=n_jobs, **kwargs)

    def grid_fit(self,
                 ecc_grid,
                 polar_grid,
                 size_grid,
                 n_grid=[1]):
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
        n_grid : list, optional
            the default is [1]
        """
        # let the gridder create the timecourses
        self.gridder.create_grid_predictions(ecc_grid=ecc_grid,
                                             polar_grid=polar_grid,
                                             size_grid=size_grid,
                                             n_grid=n_grid)

        # set up book-keeping to minimize memory usage.
        self.gridsearch_r2 = np.zeros(self.n_units)
        self.best_fitting_prediction = np.zeros(
            self.n_units, dtype=int)
        self.best_fitting_beta = np.zeros(self.n_units, dtype=float)
        self.best_fitting_baseline = np.zeros(
            self.n_units, dtype=float)

        for prediction_num in tqdm(range(self.gridder.predictions.shape[0])):
            # scipy implementation?
            # slope, intercept, rs, p_values, std_errs = linregress(self.predictions[:,prediction_num], self.data)
            # rsqs = rs**2
            # numpy implementation is slower?
            dm = np.vstack([np.ones_like(self.gridder.predictions[prediction_num]),
                            self.gridder.predictions[prediction_num]]).T
            (intercept, slope), residual, _, _ = sp.linalg.lstsq(
                dm.astype('float32'), self.data.T.astype('float32'))
            rsqs = ((1 - residual / (self.n_timepoints * self.data_var)))

            improved_fits = rsqs > self.gridsearch_r2
            # fill in the improvements
            self.best_fitting_prediction[improved_fits] = prediction_num
            self.gridsearch_r2[improved_fits] = rsqs[improved_fits]
            self.best_fitting_baseline[improved_fits] = intercept[improved_fits]
            self.best_fitting_beta[improved_fits] = slope[improved_fits]

        self.gridsearch_params = np.array([
            self.gridder.xs.ravel()[self.best_fitting_prediction],
            self.gridder.ys.ravel()[self.best_fitting_prediction],
            self.gridder.sizes.ravel()[self.best_fitting_prediction],
            self.best_fitting_beta,
            self.best_fitting_baseline,
            self.gridder.ns.ravel()[self.best_fitting_prediction],
            self.gridsearch_r2
        ]).T

    def iterative_fit(self,
                      rsq_threshold,
                      verbose=False,
                      gridsearch_params=None,
                      args={}):
        if gridsearch_params is None:
            assert hasattr(
                self, 'gridsearch_params'), 'First use self.grid_fit, or provide grid search parameters!'
        else:
            self.gridsearch_params = gridsearch_params

        if not self.fit_css:  # if we don't want to fit the n, we take it out of the parameters
            parameter_mask = np.arange(self.gridsearch_params.shape[-1] - 2)
        else:
            parameter_mask = np.arange(self.gridsearch_params.shape[-1] - 1)

        self.rsq_mask = self.gridsearch_params[:, -1] > rsq_threshold

        # create output array, knowing that iterative search adds rsq (+1)
        self.iterative_search_params = np.zeros(
            (self.n_units, len(parameter_mask) + 1))
        iterative_search_params = Parallel(self.n_jobs, verbose=verbose, prefer='threads')(
            delayed(iterative_search)(self.gridder,
                                      data,
                                      grid_pars,
                                      args=args,
                                      verbose=verbose,
                                      bounds=self.bounds,
                                      gradient_method=self.gradient_method)
            for (data, grid_pars) in zip(self.data[self.rsq_mask], self.gridsearch_params[self.rsq_mask][:, parameter_mask]))
        self.iterative_search_params[self.rsq_mask] = np.array(
            iterative_search_params)


class Norm_Iso2DGaussianFitter(Iso2DGaussianFitter):
    """Norm_Iso2DGaussianFitter

    Class that implements a grid fit on a two-dimensional isotropic
    Gaussian pRF model, leveraging a Gridder object.
    The gridder result is used as starting guess to fit the Normalization model 
    with an iterative fitting procedure

    """

    def iterative_fit(self,
                      rsq_threshold,
                      verbose=False,
                      gridsearch_params=None,
                      args={}):

        if 'bounds' not in self.__dict__:
            print("Please specify bounds on parameters")
            raise IOError

        if gridsearch_params is None:
            assert hasattr(
                self, 'gridsearch_params'), 'First use self.grid_fit, or provide grid search parameters!'
        else:
            self.gridsearch_params = gridsearch_params

        if self.gridsearch_params.shape[-1] < 8:
            # here I inject starting values for normalization model extra
            # parameters

            # neural baseline
            self.gridsearch_params = np.insert(
                self.gridsearch_params, 5, 1.0, axis=-1)
            # surround amplitude
            self.gridsearch_params = np.insert(
                self.gridsearch_params, 6, 1.0, axis=-1)  # self.gridsearch_params[:,3]/2
            # surround size
            self.gridsearch_params = np.insert(
                self.gridsearch_params, 7, 1.0, axis=-1)
            # surround baseline
            self.gridsearch_params = np.insert(
                self.gridsearch_params, 8, 1.0, axis=-1)

        # take exponent and rsq out of the parameters
        parameter_mask = np.arange(self.gridsearch_params.shape[-1] - 2)

        self.rsq_mask = self.gridsearch_params[:, -1] > rsq_threshold

        # create output array, knowing that iterative search adds rsq (+1)
        self.iterative_search_params = np.zeros(
            (self.n_units, len(parameter_mask) + 1))
        iterative_search_params = Parallel(self.n_jobs, verbose=verbose, prefer='threads')(
            delayed(iterative_search)(self.gridder,
                                      data,
                                      grid_pars,
                                      args=args,
                                      verbose=verbose,
                                      bounds=self.bounds,
                                      gradient_method=self.gradient_method)
            for (data, grid_pars) in zip(self.data[self.rsq_mask],
                                         self.gridsearch_params[self.rsq_mask][:, parameter_mask]))
        self.iterative_search_params[self.rsq_mask] = np.array(
            iterative_search_params)


class DoG_Iso2DGaussianFitter(Iso2DGaussianFitter):
    """DoG_Iso2DGaussianFitter

    Class that implements a grid fit on a two-dimensional isotropic
    difference of Gaussians pRF model, leveraging a Gridder object.
    The gridder result is used as starting guess to fit the DoG model 
    with an iterative fitting procedure

    """

    def iterative_fit(self,
                      rsq_threshold,
                      verbose=False,
                      gridsearch_params=None,
                      args={}):

        if 'bounds' not in self.__dict__:
            print("Please specify bounds on parameters")
            raise IOError

        if gridsearch_params is None:
            assert hasattr(
                self, 'gridsearch_params'), 'First use self.grid_fit, or provide grid search parameters!'
        else:
            self.gridsearch_params = gridsearch_params

        if self.gridsearch_params.shape[-1] < 8:
            # here I inject starting values for DoG model extra parameters

            # surround amplitude
            self.gridsearch_params = np.insert(
                self.gridsearch_params, 5, 0.01, axis=-1)
            # surround size
            self.gridsearch_params = np.insert(
                self.gridsearch_params, 6, 200, axis=-1)

        # take exponent and rsq out of the parameters
        parameter_mask = np.arange(self.gridsearch_params.shape[-1] - 2)

        self.rsq_mask = self.gridsearch_params[:, -1] > rsq_threshold

        # create output array, knowing that iterative search adds rsq (+1)
        self.iterative_search_params = np.zeros(
            (self.n_units, len(parameter_mask) + 1))
        iterative_search_params = Parallel(self.n_jobs, verbose=verbose, prefer='threads')(
            delayed(iterative_search)(self.gridder,
                                      data,
                                      grid_pars,
                                      args=args,
                                      verbose=verbose,
                                      bounds=self.bounds,
                                      gradient_method=self.gradient_method)
            for (data, grid_pars) in zip(self.data[self.rsq_mask],
                                         self.gridsearch_params[self.rsq_mask][:, parameter_mask]))
        self.iterative_search_params[self.rsq_mask] = np.array(
            iterative_search_params)
