import numpy as np
import os
import imageio


class PRFStimulus2D(object):
    """PRFStimulus2D

    Minimal visual 2-dimensional pRF stimulus class, 
    which takes an input design matrix and sets up its real-world dimensions.

    """

    def __init__(self,
                 screen_size_cm,
                 screen_distance_cm,                
                 TR,
                 design_matrix=None,
                 screenshot_path=None,
                 n_pix=40,
                 *args, **kwargs):
        """__init__


        Parameters
        ----------
        screen_size_cm : float
            size of screen in centimeters
        screen_distance_cm : float
            eye-screen distance in centimeters
        screenshots_path: string
            if provided, construct a design_matrix from a sequence of numbered PNG images found in this path
        design_matrix : numpy.ndarray
            an N by t matrix, where N is [x, x]. 
            represents a square screen evolving over time (time is last dimension)

        """
        self.screen_size_cm = screen_size_cm
        self.screen_distance_cm = screen_distance_cm
        
        if design_matrix is not None:
            self.design_matrix = design_matrix
        else:
            if screenshot_path is None:
                print("Need to specify either design matrix or screenshot path!")
                raise IOError
            else:
                image_list = os.listdir(screenshot_path)
                self.design_matrix=np.zeros((n_pix,n_pix,len(image_list)))
                for image_file in image_list:
                    #assuming last three numbers before .png are the screenshot number
                    img_number = int(image_file[-7:-4]) - 1
                    #subtract one to start from zero
                    img = imageio.imread(os.path.join(screenshot_path,image_file))
                    #make it square and downsample
                    if img.shape[0]!=img.shape[1]:
                        offset=int((img.shape[1]-img.shape[0])/2)
                        img=img[:,offset:(offset+img.shape[0])]
                    
                    downsampling_constant = int(img.shape[1]/n_pix)
                    downsampled_img = img[::downsampling_constant,::downsampling_constant]
                    
                    #binarize image 
                    #assumes standard RGB255 format; assumes only colors present in image are black, white, grey, red, green.
                    self.design_matrix[:,:,img_number][np.where(((downsampled_img[:,:,0] == 0) & (downsampled_img[:,:,1] == 0)) | ((downsampled_img[:,:,0] == 255) & (downsampled_img[:,:,1] == 255)))] = 1
                    
                    
                
                    

            
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


DMFromScreenshots