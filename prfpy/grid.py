import numpy as np
from .rf import *   # import all possible RF shapes
from .timecourse import *
from hrf_estimation import hrf


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

    # define the prerequisite rf structure
    rf_function = gauss2D_iso_cart

    def __init__(self, stimulus, hrf=None, **kwargs):
        super(Iso2DGaussianGridder, Gridder).__init__(stimulus)
        self.__dict__.update(kwargs)

        if hrf == None:  # for use with standard fMRI
            self.hrf = hrf.spmt
        elif hrf == 'direct':  # for use with anything like eCoG with instantaneous irf
            self.hrf = np.array([1])
        # some specific hrf with spm basis set
        elif ((type(hrf) == list) or (type(hrf) == np.ndarray)) and len(hrf) == 3:
            hrf_times = np.linspace(0, 40, 40/self.stimulus.TR, endpoint=False)
            self.hrf = np.array([hrf[0] * hrf.spmt(hrf_times),
                                 hrf[1] * hrf.dspmt(hrf_times),
                                 hrf[2] * hrf.ddspmt(hrf_times)]).sum(axis=0)
        # some specific hrf already defined at the TR (!)
        elif type(hrf) == np.ndarray and len(hrf) > 3:
            self.hrf = hrf

        self.convolved_design_matrix = convolve_stimulus_dm(
            stimulus.design_matrix, hrf=self.hrf)

    def setup_grid(self, ecc_grid=None, polar_grid=None, size_grid=None, n_grid=[1]):
        """setup_grid

        setup_grid sets up the parameters that span the grid. this assumes both baseline 
        and betas (amplitudes) are not part of this grid and will fall out of the GLM

        Parameters
        ----------
        ecc_grid : list or numpy.ndarray
            contains all the settings for the ecc dimension of the grid
        polar_grid : list or numpy.ndarray
            contains all the settings for the polar dimension of the grid
        size_grid : list or numpy.ndarray
            contains all the settings for the size dimension of the grid
        n_grid : list or numpy.ndarray
            contains all the settings for the normalization dimension of the grid

        """
        assert ecc_grid != None and polar_grid != None and size_grid != None, \
            "please fill in all spatial grids"

        self.eccs, self.polars, self.sizes, self.ns = np.meshgrid(
            ecc_grid, polar_grid, size_grid, n_grid)
        self.xs, self.ys = np.sin(self.polars) * \
            self.eccs, np.cos(self.polars) * self.eccs

    def create_rfs(self):
        """create_rfs

        creates rfs for the grid

        """
        assert hasattr(self, 'xs'), "please set up the grid first"
        self.grid_rfs = Iso2DGaussianGridder.rf_function(
            x=self.stimulus.x_coordinates[..., np.newaxis],
            y=self.stimulus.y_coordinates[..., np.newaxis],
            mu=np.array([self.xs, self.ys]).reshape((-1, 2)).T,
            sigma=self.sizes.ravel())
        # won't have to perform exponentiation of all ns are 1
        if np.unique(self.ns) == 1:
            self.grid_rfs **= self.ns.ravel()

    def stimulus_times_prfs(self):
        """stimulus_times_prfs

        creates timecourses for each of the rfs in self.grid_rfs

        """
        assert hasattr(self, 'grid_rfs'), "please create the rfs first"
        self.predictions = stimulus_through_prf(
            self.grid_rfs, self.convolved_design_matrix)
