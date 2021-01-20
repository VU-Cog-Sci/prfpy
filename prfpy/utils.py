import cortex
from scipy import stats
import numpy as np


def basic_plot(dat,vmax,subject='fsaverage',vmin=0,rois=False,colorbar=False,cmap='plasma',ax=None):
    light=cortex.Vertex(dat,subject=subject, vmin=vmin, vmax=vmax,cmap=cmap)
    mfig=cortex.quickshow(light,with_curvature=True,with_rois=rois,with_colorbar=colorbar,fig=ax)

class subsurface(object):
    
    """subsurface_generator
        Use pycortex utilities to generate info about sub-surfaces.
        
    """

    def __init__(self,cx_sub,boolmasks,surftype='fiducial'):
        
        """__init__
        Parameters
        ----------
        cx_sub : The name of the cx subject (string). This is used to get surfaces from the pycx database.
        boolmasks: A list of boolean arrays that define the vertices that correspond to the ROI one wants to make a subsurface from [left hemisphere, right hemisphere].
        surftype: The surface (default = fiducial).
        """
        
        self.cx_sub=cx_sub
        self.surftype=surftype
        self.boolmasks=boolmasks
        self.mask=np.concatenate([self.boolmasks[0],self.boolmasks[1]]).astype(int) # Put the mask into int format for plotting.
        

        
    def create(self):
        self.get_surfaces()
        self.generate()
        self.get_geometry()
        self.pad_distance_matrices()
        
    def get_surfaces(self):
        
        """get_surfaces
        Accesses the pycortex database to return the subject surfaces (left and right).
        """
        
        self.surfaces = [cortex.polyutils.Surface(*d)
        for d in cortex.db.get_surf(self.cx_sub, self.surftype)]
                
    def generate(self):
        
        """generate
        Use the masks defined in boolmasks to define subsurfaces.
        """
        
        print('Generating subsurfaces')
        self.subsurface_L = self.surfaces[0].create_subsurface(vertex_mask=self.boolmasks[0]) # Create sub-surfaces.
        self.subsurface_R = self.surfaces[1].create_subsurface(vertex_mask=self.boolmasks[1])
        
        # Get the whole-brain indices for those vertices contained in the subsurface.
        self.subsurface_verts_L=np.where(self.subsurface_L.subsurface_vertex_map!=stats.mode(self.subsurface_L.subsurface_vertex_map)[0][0])[0]
        self.subsurface_verts_R=np.where(self.subsurface_R.subsurface_vertex_map!=stats.mode(self.subsurface_R.subsurface_vertex_map)[0][0])[0]+self.subsurface_L.subsurface_vertex_map.shape[-1]
        

    def get_geometry(self):
        
        """get_geometry
        Returns geometric info about the sub-surfaces. Computes geodesic distances from each point of the sub-surface.
        """
        
        # Assign some variables to determine where the boundary between the hemispheres is. 
        self.leftlim=np.max(self.subsurface_verts_L)
        self.n_leftverts=self.subsurface_verts_L.shape[-1]
        
        self.subsurface_verts=np.concatenate([self.subsurface_verts_L,self.subsurface_verts_R])
        
        # Make the distance x distance matrix.
        ldists,rdists=[],[]

        print('Creating distance by distance matrices')
        
        for i in range(len(self.subsurface_verts_L)):
            ldists.append(self.subsurface_L.geodesic_distance([i]))
        self.dists_L=np.array(ldists)
        
        for i in range(len(self.subsurface_verts_R)):
            rdists.append(self.subsurface_R.geodesic_distance([i]))
        self.dists_R=np.array(rdists)
        
        self.surfaces,self.subsurface_L,self.subsurface_R=None,None,None # Get rid of these as they are harmful for pickling. We no longer need them.
        
        
    def pad_distance_matrices(self,padval=np.Inf):
        
        """pad_distance_matrices
        Pads the distance matrices so that distances to the opposite hemisphere are np.inf
        Stack them on top of each other so they will have the same size as the design matrix
        """
        
        padL=np.pad(self.dists_L, ((0, 0), (0, self.dists_R.shape[-1])),constant_values=np.Inf) # back pad the left hem .
        padR=np.pad(self.dists_R, ((0, 0), (self.dists_L.shape[-1],0)),constant_values=np.Inf) # front pad the right hem.
        
        self.distance_matrix=np.vstack([padL,padR]) # Now stack.
        
    def elaborate(self):
        
        """elaborate
        Prints information about the created subsurfaces.

        """
            
        print(f"Maximum distance across subsurface: {np.max(self.dists_L)} mm")
        print(f"Vertices in left hemisphere: {self.dists_L.shape[-1]}")
        print(f"Vertices in right hemisphere: {self.dists_R.shape[-1]}")
        
    def show(self,cmap='gist_ncar'):
        
        """show
        Plots the subsurfaces.
        """
        
        basic_plot(self.mask,vmax=1,subject=self.cx_sub,cmap=cmap)
    

def masked_plot(dat,noiseinds,vmax,subject='fsaverage',vmin=0,rois=False,colorbar=False,cmap='plasma',ax=None):
    maskdat=np.copy(dat)
    maskdat[noiseinds]=np.nan
    basic_plot(maskdat,vmax,subject,vmin,rois,colorbar,cmap,ax)
    
    
def alpha_plot(dat,dat2,vmax,subject='fsaverage',vmin=0,rois=False,colorbar=False,cmap='plasma_alpha',ax=None):
    light=cortex.Vertex2D(dat,dat2,subject=subject, vmin=vmin, vmax=vmax,vmin2=0,vmax2=np.nanmax(dat2),cmap=cmap)
    mfig=cortex.quickshow(light,with_curvature=True,with_rois=rois,with_colorbar=colorbar,fig=ax)

    
def zoomed_plot(dat,vmax,ROI,hem,subject='hcp_999999',vmin=0,rois=False,colorbar=False,cmap='plasma',ax=None):
    basic_plot(dat,vmax,subject,vmin,rois,colorbar,cmap,ax)
    zoom_to_roi(subject,ROI,hem)
    
def zoom_to_roi(subject, roi, hem, margin=35.0):
    roi_verts = cortex.get_roi_verts(subject, roi)[roi]
    roi_map = cortex.Vertex.empty(subject)
    roi_map.data[roi_verts] = 1

    (lflatpts, lpolys), (rflatpts, rpolys) = cortex.db.get_surf(subject, "flat",
                                                                nudge=True)
    sel_pts = dict(left=lflatpts, right=rflatpts)[hem]
    roi_pts = sel_pts[np.nonzero(getattr(roi_map, hem))[0],:2]

    xmin, ymin = roi_pts.min(0) - margin
    xmax, ymax = roi_pts.max(0) + margin
    
    
    plt.axis([xmin, xmax, ymin, ymax])
    return
