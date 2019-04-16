import math
import numpy as np

import keras
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LocallyConnected1D
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from keras.utils.vis_utils import plot_model


class Gaussian2D_isoCart_pRF_Sequence(keras.utils.Sequence):
    """Gaussian2D_isoCart_pRF_Sequence

    class to generate pRF model timecourses, inherits from keras Sequence 

    """

    def __init__(self, n_timepoints, batch_size, gridder, ecc_range, size_range, beta_range, baseline_range, n_range, grid_size, drift_ranges=[[0, 0]], noise_level=0, noise_ar=(1, 0.4)):
        self.n_timepoints = n_timepoints
        self.batch_size = batch_size
        self.gridder = gridder
        self.ecc_range = ecc_range
        self.size_range = size_range
        self.beta_range = beta_range
        self.baseline_range = baseline_range
        self.n_range = n_range
        self.grid_size = grid_size
        self.drift_ranges = drift_ranges
        self.noise_level = noise_level
        self.noise_ar = noise_ar

        self.epoch_nr = 0

        self.gridder.create_grid_predictions(
            ecc_grid=np.linspace(
                self.ecc_range[0], self.ecc_range[1], self.grid_size),
            polar_grid=np.linspace(0, 2*np.pi, self.grid_size),
            size_grid=np.linspace(
                self.size_range[0], self.size_range[1], self.grid_size),
            n_grid=np.linspace(
                self.n_range[0], self.n_range[1], self.grid_size),
        )

    def on_epoch_end(self):
        """creates new pRF models based on initial parameters"""
        self.parameters = np.array([
            self.gridder.xs.ravel(),
            self.gridder.ys.ravel(),
            self.gridder.sizes.ravel(),
            self.beta_range[0] +
            (self.beta_range[1]-self.beta_range[0]) *
            np.random.rand(gridder.predictions.shape[0]),
            self.baseline_range[0] +
            (self.baseline_range[1]-self.baseline_range[0]) *
            np.random.rand(gridder.predictions.shape[0]),
            self.gridder.ns.ravel()
        ]).T

        # implement the random beta and baselines
        self.predictions = self.parameters[:, 4] + self.parameters[:, 3] * gridder.predictions + \
            gridder.create_drifts_and_noise(drift_ranges=self.drift_ranges,
                                            noise_ar=noise_ar,
                                            noise_amplitude=self.noise_level)

        self.epoch_nr += 1

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(self.predictions.shape[0] / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        X = self.predictions[index*self.batch_size:(index+1)*self.batch_size]
        y = self.parameters[index*self.batch_size:(index+1)*self.batch_size]

        return X, y


def create_cnn(n_timepoints, n_parameters, loss='mse', optimizer='adam', metrics=['mae']):
    """create_cnn

    creates and compiles a cnn by adding a nr of Conv1D layers
    that depends on the n_timepoints. These are topped off with a pair of 
    Dense layers that end up leading to the output parameters.

    [description]

    Parameters
    ----------
    n_timepoints : int
        number of timepoints in the data
    n_parameters : int
        number of parameters in the pRF model to be learned
    loss : str, optional
        Keras loss function (the default is 'mse', which is good for the learning of continuous outputs)
    optimizer : str, optional
        Keras optimizer function (the default is 'adam', which is good for the learning of continuous outputs)
    metrics : list, optional
        Keras metrics used for model performance quantification 
        (the default is ['mae'], which which is good for the learning of continuous outputs)

    Returns
    -------
    Sequential
        Compiled Keras model
    """
    n_Conv1D_layers = n_timepoints//32

    model = Sequential()
    # first convolutional layer here
    model.add(Conv1D(filters=n_timepoints//2,
                     kernel_size=4, input_shape=(n_timepoints, 1)))
    model.add(MaxPooling1D(pool_size=2))

    # loop over nr of required layers
    for l in range(2, n_Conv1D_layers+1):
        n_filters = n_timepoints//(2**l)
        kernel_size = 2**l
        model.add(Conv1D(filters=n_filters, kernel_size=kernel_size,
                         input_shape=(n_timepoints, 1)))
        model.add(MaxPooling1D(pool_size=kernel_size/2))

    model.add(Flatten())
    model.add(Dense(n_parameters*2, kernel_initializer='uniform'))
    model.add(Dense(n_parameters))

    model.compile(loss='mse',
                  optimizer='nadam',
                  metrics=['mse'])

    return model
