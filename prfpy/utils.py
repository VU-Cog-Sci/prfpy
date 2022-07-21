import cortex
from scipy import stats
import numpy as np

# Imports for HRF
from nilearn.glm.first_level.hemodynamic_models import spm_hrf, spm_time_derivative, spm_dispersion_derivative


class Subsurface(object):

    """subsurface
        This is a utility that uses pycortex for making sub-surfaces for CF fitting.

    """

    def __init__(self, cx_sub, boolmasks, surftype='fiducial'):
        """__init__
        Parameters
        ----------
        cx_sub : The name of the cx subject (string). This is used to get surfaces from the pycx database.
        boolmasks: A list of boolean arrays that define the vertices that correspond to the ROI one wants to make a subsurface from [left hem, right hem].
        surftype: The surface (default = fiducial).
        """

        self.cx_sub = cx_sub
        self.surftype = surftype
        self.boolmasks = boolmasks
        # Put the mask into int format for plotting.
        self.mask = np.concatenate(
            [self.boolmasks[0], self.boolmasks[1]]).astype(int)

    def create(self):
        """get_surfaces
        Function that creates the subsurfaces.
        """

        self.get_surfaces()
        self.generate()
        self.get_geometry()
        self.pad_distance_matrices()

    def get_surfaces(self):
        """get_surfaces
        Accesses the pycortex database to return the subject surfaces (left and right).

        Returns
        -------
        subsurface_L, subsurface_R: A pycortex subsurfaces classes for each hemisphere (These are later deleted by 'get_geometry', but can be re-created with a call to this function).
        self.subsurface_verts_L,self.subsurface_verts_R : The whole brain indices of each vertex in the subsurfaces.

        """

        self.surfaces = [cortex.polyutils.Surface(*d)
                         for d in cortex.db.get_surf(self.cx_sub, self.surftype)]

    def generate(self):
        """generate
        Use the masks defined in boolmasks to define subsurfaces.
        """

        print('Generating subsurfaces')
        # Create sub-surface, left hem.
        self.subsurface_L = self.surfaces[0].create_subsurface(
            vertex_mask=self.boolmasks[0])
        # Create sub-surface, right hem.
        self.subsurface_R = self.surfaces[1].create_subsurface(
            vertex_mask=self.boolmasks[1])

        # Get the whole-brain indices for those vertices contained in the subsurface.
        self.subsurface_verts_L = np.where(self.subsurface_L.subsurface_vertex_map != stats.mode(
            self.subsurface_L.subsurface_vertex_map)[0][0])[0]
        self.subsurface_verts_R = np.where(self.subsurface_R.subsurface_vertex_map != stats.mode(
            self.subsurface_R.subsurface_vertex_map)[0][0])[0]+self.subsurface_L.subsurface_vertex_map.shape[-1]

    def get_geometry(self):
        """get_geometry
        Calculates geometric info about the sub-surfaces. Computes geodesic distances from each point of the sub-surface.

        Returns
        -------
        dists_L, dists_R: Matrices of size n vertices x n vertices that describes the distances between all vertices in each hemisphere of the subsurface.
        subsurface_verts: The whole brain indices of each vertex in the subsurface.
        leftlim: The index that indicates the boundary between the left and right hemisphere. 
        """

        # Assign some variables to determine where the boundary between the hemispheres is.
        self.leftlim = np.max(self.subsurface_verts_L)
        self.subsurface_verts = np.concatenate(
            [self.subsurface_verts_L, self.subsurface_verts_R])

        # Make the distance x distance matrix.
        ldists, rdists = [], []

        print('Creating distance by distance matrices')

        for i in range(len(self.subsurface_verts_L)):
            ldists.append(self.subsurface_L.geodesic_distance([i]))
        self.dists_L = np.array(ldists)

        for i in range(len(self.subsurface_verts_R)):
            rdists.append(self.subsurface_R.geodesic_distance([i]))
        self.dists_R = np.array(rdists)

        # Get rid of these as they are harmful for pickling. We no longer need them.
        self.surfaces, self.subsurface_L, self.subsurface_R = None, None, None

    def pad_distance_matrices(self, padval=np.Inf):
        """pad_distance_matrices
        Pads the distance matrices so that distances to the opposite hemisphere are np.inf
        Stack them on top of each other so they will have the same size as the design matrix


        Returns
        -------
        distance_matrix: A matrix of size n vertices x n vertices that describes the distances between all vertices in the subsurface.
        """

        # Pad the right hem with np.inf.
        padL = np.pad(
            self.dists_L, ((0, 0), (0, self.dists_R.shape[-1])), constant_values=np.Inf)
        # pad the left hem with np.inf..
        padR = np.pad(
            self.dists_R, ((0, 0), (self.dists_L.shape[-1], 0)), constant_values=np.Inf)

        self.distance_matrix = np.vstack([padL, padR])  # Now stack.

    def elaborate(self):
        """elaborate
        Prints information about the created subsurfaces.

        """

        print(
            f"Maximum distance across left subsurface: {np.max(self.dists_L)} mm")
        print(
            f"Maximum distance across right subsurface: {np.max(self.dists_R)} mm")
        print(f"Vertices in left hemisphere: {self.dists_L.shape[-1]}")
        print(f"Vertices in right hemisphere: {self.dists_R.shape[-1]}")


def squaresign(vec):
    """squaresign
        Raises something to a power in a sign-sensive way.
        Useful for if dot products happen to be negative.
    """
    vec2 = (vec**2)*np.sign(vec)
    return vec2
