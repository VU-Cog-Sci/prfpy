import numpy as np
import scipy as sp
from scipy.optimize import fmin_powell
import bottleneck as bn
from tqdm import tqdm

from joblib import Parallel, delayed

from .grid import Iso2DGaussianGridder


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
    return bn.nansum((data-objective_function(*list(parameters), **args))**2)


def iterative_search(gridder, data, grid_params, args, verbose=True):
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
    output = fmin_powell(error_function, grid_params, xtol=0.001, ftol=0.001,
                         args=(args, data, gridder.return_single_prediction),
                         full_output=True, disp=verbose)
    return np.r_[output[0],  1 - (output[1]/(len(data)*data.var()))]


class Fitter:
    def __init__(self, data, gridder, n_jobs=1, **kwargs):
        self.data = data
        self.gridder = gridder
        self.n_jobs = n_jobs
        self.__dict__.update(kwargs)

        self.n_units = np.prod(self.data.shape[:-1])
        self.n_timepoints = self.data.shape[-1]

        # immediately convert nans to nums
        self.data = np.nan_to_num(data)
        self.data_var = self.data.var(axis=-1)


class Iso2DGaussianFitter(Fitter):
    """Iso2DGaussianFitter

    Class that implements the different fitting methods, 
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
                dm, self.data.T)
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
            self.gridder.ns.ravel()[self.best_fitting_prediction]
        ]).T

    def iterative_fit(self,
                      rsq_threshold,
                      verbose=False,
                      args={}):
        assert hasattr(self, 'gridsearch_params'), 'First use self.fit_grid!'

        if not self.fit_css:  # if we don't want to fit the n, we take it out of the parameters
            parameter_mask = np.arange(self.gridsearch_params.shape[-1]-1)
        else:
            parameter_mask = np.arange(self.gridsearch_params.shape[-1])

        self.rsq_mask = self.gridsearch_r2 > rsq_threshold

        # create output array, knowing that iterative search adds rsq (+1)
        self.iterative_search_params = np.zeros(
            (self.n_units, len(parameter_mask)+1))
        iterative_search_params = Parallel(self.n_jobs, verbose=verbose)(
            delayed(iterative_search)(self.gridder,
                                      data,
                                      grid_pars,
                                      args=args, verbose=verbose)
            for (data, grid_pars) in zip(self.data[self.rsq_mask], self.gridsearch_params[self.rsq_mask][:, parameter_mask]))
        self.iterative_search_params[self.rsq_mask] = np.array(
            iterative_search_params)
