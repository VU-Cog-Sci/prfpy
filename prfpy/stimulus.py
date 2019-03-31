import numpy as np

class PRFStimulus(object):
    """PRFStimulus
    
    Minimal pRF stimulus class, which takes an input design matrix, and sets up its real-world dimensions.
    
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
        if self.design_matrix.shape[0] != self.design_matrix.shape[1]:
            raise ValueError # need the screen to be square
        self.TR = TR

        self.screen_size_degrees = 2.0 * np.degrees(np.arctan(self.screen_size_cm/(2.0*self.screen_distance_cm)))

        oneD_grid = np.linspace(-self.screen_size_degrees/2, 
                                self.screen_size_degrees/2, 
                                self.design_matrix.shape[0], 
                                endpoint=True)
        self.x_coordinates, self.y_coordinates = np.meshgrid(oneD_grid, oneD_grid)


        