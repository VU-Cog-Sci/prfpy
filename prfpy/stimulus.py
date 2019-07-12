import numpy as np


class PRFStimulus2D(object):
    """PRFStimulus2D

    Minimal visual 2-dimensional pRF stimulus class, 
    which takes an input design matrix and sets up its real-world dimensions.

    """

    def __init__(self,
                 screen_size_cm,
                 screen_distance_cm,
                 design_matrix,
                 TR,
                 *args, **kwargs):
        """__init__


        Parameters
        ----------
        screen_size_cm : float
            size of screen in centimeters
        screen_distance_cm : float
            eye-screen distance in centimeters
        design_matrix : numpy.ndarray
            an N by t matrix, where N is [x, x]. 
            represents a square screen evolving over time (time is last dimension)

        """
        self.screen_size_cm = screen_size_cm
        self.screen_distance_cm = screen_distance_cm
        self.design_matrix = design_matrix
        if len(self.design_matrix.shape) >= 3 and self.design_matrix.shape[0] != self.design_matrix.shape[1]:
            raise ValueError  # need the screen to be square
        self.TR = TR

        self.screen_size_degrees = 2.0 * \
            np.degrees(np.arctan(self.screen_size_cm /
                                 (2.0*self.screen_distance_cm)))

        oneD_grid = np.linspace(-self.screen_size_degrees/2,
                                self.screen_size_degrees/2,
                                self.design_matrix.shape[0],
                                endpoint=True)
        self.x_coordinates, self.y_coordinates = np.meshgrid(
            oneD_grid, oneD_grid)
        self.complex_coordinates = self.x_coordinates + self.y_coordinates * 1j
        self.ecc_coordinates = np.abs(self.complex_coordinates)
        self.polar_coordinates = np.angle(self.complex_coordinates)
        self.max_ecc = np.max(self.ecc_coordinates)

        # construct a standard mask based on standard deviation over time
        self.mask = np.std(design_matrix, axis=-1) != 0


class PRFStimulus1D(object):
    """PRFStimulus1D

    Minimal visual 1-dimensional pRF stimulus class, 
    which takes an input design matrix and sets up its real-world dimensions.

    """

    def __init__(self,
                 design_matrix,
                 mapping,
                 TR,
                 *args, **kwargs):
        """__init__


        Parameters
        ----------
        design_matrix : numpy.ndarray
            a 2D matrix (M by t). 
            represents inputs in an encoding space evolving over time (time is last dimension)
        mapping : numpy.ndarray, np.float
            for each of the columns in design_matrix, the value in the encoding dimension
            for example, in a numerosity experiment these would be the numerosity of presented stimuli
        TR : float
            repetition time of the design matrix

        """
        self.design_matrix = design_matrix
        self.mapping = mapping
        self.TR = TR
