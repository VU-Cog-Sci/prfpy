import numpy as np
import scipy as sp
from scipy.optimize import fminpowell
import bottleneck as bn
from tqdm import tqdm

from joblib import Parallel, delayed

from .grid import Iso2DGaussianGridder

def error_function(parameters, args, data, objective_function, verbose):
    """Generic error function.
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
        proceduces a model time-series.
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
        `return_single_timecourse` method
    data : 1D numpy.ndarray
        the data to fit, same dimensions as are returned by 
        Gridder's `return_single_timecourse` method
    grid_params : tuple [float]
        initial values for the fit
    args : dictionary, arguments to gridder.return_single_timecourse that 
        are not optimized
    verbose : bool, optional
        whether to have fminpowell puke everything out. 
    
    Returns
    -------
    2-tuple
        first element: parameter values,
        second element: rsq value
    """
    output = fmin_powell(error_function, grid_params,
                         args=(args, data, gridder.return_single_timecourse, verbose),
                         full_output=True, disp=verbose)
    return np.r_[output[0],  1 - (output[1]/(len(data)*data.var()))]

class Fitter(object):
    def ___init___(self, data, gridder, n_jobs, **kwargs):
        self.data = data
        self.gridder = gridder
        self.n_jobs = n_jobs
        self.__dict__.update(kwargs)

        self.n_units = np.prod(self.data.shape[:-1])
        self.n_timepoints = self.data.shape[-1]

class Iso2DGaussianFitter(Fitter):
    """Iso2DGaussianFitter
    
    Class that implements the different fitting methods, 
    leveraging a Gridder object.
    
    """

    def grid_fit(self,
                ecc_grid,
                polar_grid,
                size_grid,
                n_grid=[1]):
        """setup_grid_specs
        
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
            to be filled in by user 
            (the default is [1, 1, 1], which returns [1] as array)
        """
        # let the gridder create the timecourses
        self.gridder.create_timecourses(ecc_grid=ecc_grid,
                           polar_grid=polar_grid,
                           size_grid=size_grid,
                           n_grid=n_grid)
        
        # set up book-keeping to minimize memory usage.
        self.gridsearch_r2 = np.zeros(self.n_units)
        self.best_fitting_prediction_thus_far = np.zeros(self.n_units, dtype=int)
        self.best_fitting_beta_thus_far = np.zeros(self.n_units, dtype=float)
        self.best_fitting_baseline_thus_far = np.zeros(self.n_units, dtype=float)

        prediction_params = np.ones((self.n_units, 7))*np.nan

        for prediction_num in tqdm(range(self.gridder.predictions.shape[1])):
            # scipy implementation?
            # slope, intercept, rs, p_values, std_errs = linregress(self.predictions[:,prediction_num], self.data)
            # rsqs = rs**2
            # numpy implementation is slower?
            dm = np.vstack([np.ones_like(self.gridder.predictions[:,prediction_num]),
                                        self.gridder.predictions[:,prediction_num]]).T
            (intercept, slope), residual, _, _ = sp.linalg.lstsq(dm, self.data.T, check_finite=False) 
            rsqs = ((1 - residual / (self.n_timepoints * self.data_var)))

            improved_fits = rsqs > self.gridsearch_r2
            # fill in the improvements
            self.best_fitting_prediction_thus_far[improved_fits] = prediction_num
            self.gridsearch_r2[improved_fits] = rsqs[improved_fits]
            self.best_fitting_baseline_thus_far[improved_fits] = intercept[improved_fits]
            self.best_fitting_beta_thus_far[improved_fits] = slope[improved_fits]

            self.gridsearch_params = np.array([ self.gridder.xs.ravel()[self.best_fitting_prediction_thus_far],
                                                    self.gridder.ys.ravel()[self.best_fitting_prediction_thus_far],
                                                    self.gridder.sizes.ravel()[self.best_fitting_prediction_thus_far],
                                                    self.gridder.ns.ravel()[self.best_fitting_prediction_thus_far],
                                                    self.best_fitting_beta_thus_far,
                                                    self.best_fitting_baseline_thus_far
                                                ])

    def iterative_fit(self, 
                    rsq_threshold,
                    verbose=True, 
                    args={}):
        assert hasattr(self, 'gridsearch_params'), 'First use self.fit_grid!'
        
        self.iterative_search_params = Parallel(self.n_jobs, verbose=verbose)(delayed(iterative_search)(self.gridder, 
                                                                                    data, 
                                                                                    grid_pars, 
                                                                                    args=args)
                                       for (data,grid_pars) in zip(self.data, self.gridsearch_params))
        self.iterative_search_params = np.array(self.iterative_search_params)
