import numpy as np
import scipy as sp

from .grid import Iso2DGaussianGridder


class Fitter(object):
    def ___init___(self, data, gridder, **kwargs):
        self.data = data
        self.gridder = gridder
        self.__dict__.update(kwargs)


class Iso2DGaussianFitter(Fitter):

    def setup_grid_specs(ecc_grid=ecc_grid,
                         polar_grid=polar_grid,
                         size_grid=size_grid,
                         n_grid=n_grid):
        pass

    def grid_fit(args):
        pass
