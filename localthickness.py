import numpy as np
import skimage.morphology
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.colors

#%% FUNCTIONS FOR COMPUTING LOCAL THICKNESS IN 2D AND 3D

def local_thickness(B, mask=None):
    """
    Computes local thickness in 2D or 3D.
    @author: abda@dtu.dk, vand@dtu.dk
    Arguments: B - binary 2D or 3D image.
    Returns: Local thickness of the same size as B.
    """
    # distance field
    df = scipy.ndimage.distance_transform_edt(B)
    if mask is not None:
        df = df*mask
    # image that will be updated
    out = np.copy(df) 
    # structuring elements and weights
    elements = structuring_elements(B.ndim)
    # iteratively dilate the distance field starting with max value
    for r in range(0, int(np.max(df))):
        temp = np.zeros(B.shape)
        # weighted-dilation with a set of structuring elements
        
        for d in range(len(elements)):
            print(_tofootprint(elements[d][0]))
            temp += elements[d][1]*skimage.morphology.dilation(out, footprint=_tofootprint(elements[d][0]))
        change = out>r
        out[change] = temp[change]
    return out

def local_thickness_multiscale(B, nr_scales=3, mask=None):
    """
    Computes local thickness in 2D or 3D using multi-scale approach.
    @author: abda@dtu.dk, vand@dtu.dk
    Arguments: V - binary 2D or 3D volume, nr_scales - number of scales (default 3).
    Returns: Local thickness of the same size as B.
    """
    dim = B.shape # original image dimension
    dim_s = tuple(d//(2**(nr_scales-1)) for d in dim) # smallest (starting) image dimension
    # downscaling the volumes
    if mask is None:
        mask_s = None
    else: 
        mask_s = skimage.transform.resize(mask, dim_s, order=0, 
                        anti_aliasing_sigma=0, preserve_range=True).astype(bool)
    B_s = skimage.transform.resize(B, dim_s, order=0, anti_aliasing_sigma=0, 
                        preserve_range=True).astype(bool)
    # computing local thickness for downscaled
    out = local_thickness(B_s,mask_s)
    # free up some memery
    del B_s
    del mask_s
    # structuring elements and weights
    elements = structuring_elements(B.ndim)
    # iteratively upscale and compute one dilation in each iteration
    for i in range(0, nr_scales-1):
        dim_to = tuple(d//(2**(nr_scales-2-i)) for d in dim)
        out = 2*skimage.transform.resize(out, dim_to, order=1, preserve_range=True)
        temp = np.zeros(out.shape)
        for d in range(len(elements)):
            temp += elements[d][1]*skimage.morphology.dilation(out, footprint=_tofootprint(elements[d][0],))
        out = temp
    out *= B
    
    # mask output
    if mask is not None:
        out *= mask
    return out


def local_thickness_scaled(B, scale=0.5, mask=None):
    """
    Computes local thickness in 2D or 3D using scaled approach.
    @author: abda@dtu.dk, vand@dtu.dk
    Arguments: V - binary 2D or 3D volume, scale - downscalng factor (default 0.5).
    Returns: Local thickness of the same size as B.
    """
    dim = B.shape # original image dimension
    dim_s = tuple(int(scale*d) for d in dim) # smallest (starting) image dimension
    # downscaling the volumes, skimage order=0 is nearest-neighbor
    if mask is None:
        mask_s = None
    else: 
        mask_s = skimage.transform.resize(mask, dim_s, order=0, 
                        anti_aliasing_sigma=0, preserve_range=True).astype(bool) 
    B_s = skimage.transform.resize(B, dim_s, order=0, anti_aliasing_sigma=0, 
                        preserve_range=True).astype(bool)
    
    # computing local thickness for downscaled
    out = local_thickness(B_s,mask_s)
    
    # flow-over boundary to avoid blend across boundary, will mask later
    elements = structuring_elements(B.ndim)
    temp = np.zeros(out.shape)
    for d in range(len(elements)):
        temp += elements[d][1]*skimage.morphology.dilation(out, footprint=_tofootprint(elements[d][0]))
    out[~B_s] = temp[~B_s]
    
    # free up some memery
    del B_s
    del mask_s

    # upscale, skimage order=3 is bi-cubic
    out = (1/scale)*skimage.transform.resize(out, dim, order=1, preserve_range=True)
    out *= B    
    
    # mask output
    if mask is not None:
        out *= mask
    return out


def local_thickness_conventional(B, mask = None):
    """
    Computes local thickness in 2D or 3D using the conventional approach.
    VERY SLOW, NOT TESTED, USE WITH CAUTION!!!!
    THIS IS JUST FOR COMPARISON!!
    @author: abda@dtu.dk, vand@dtu.dk
    Arguments: B - binary 2D or 3D image.
    Returns: Local thickness of the same size as B.
    """
    # distance field
    df = scipy.ndimage.distance_transform_edt(B)
    if mask is not None:
        df = df*mask
    # image that will be updated
    out = np.copy(df) 
    # iteratively dilate the distance field starting with max value
    for r in range(1, int(np.max(df))+1):
        if B.ndim==2:
            selem = skimage.morphology.disk(r)
        elif B.ndim==3:
            selem = skimage.morphology.ball(r)
        temp = skimage.morphology.dilation(df*(df>=r), footprint=_tofootprint(selem))
        change = temp>r    
        out[change] = temp[change]
    out *= B
    if mask is not None:
        out *= mask
    return out       


def structuring_elements(d):
    """
    Structuring elements for 2D or 3D local thickness.
    Arguments: d - dimensionality (2 or 3)
    """
    if d==2:
        # structuring elements for 2D local thickness
        selem_plus = [[0,1,0],[1,1,1],[0,1,0]]
        selem_cross = [[1,0,1],[0,1,0],[1,0,1]]
        # weights
        w_plus = np.sqrt(2)/(1+np.sqrt(2))
        w_cross = 1/(1+np.sqrt(2))
        return((selem_plus, w_plus), (selem_cross, w_cross))
    elif d==3:
        # structuring elements for 3D local thickness
        selem_plus = [[[0,0,0],[0,1,0],[0,0,0]], [[0,1,0],[1,1,1],[0,1,0]], 
                      [[0,0,0],[0,1,0],[0,0,0]]]
        selem_cross = [[[0,1,0],[1,0,1],[0,1,0]], [[1,0,1],[0,1,0],[1,0,1]], 
                       [[0,1,0],[1,0,1],[0,1,0]]]
        selem_star = [[[1,0,1],[0,0,0],[1,0,1]], [[0,0,0],[0,1,0],[0,0,0]], 
                      [[1,0,1],[0,0,0],[1,0,1]]]
        # weights
        w_plus = np.sqrt(6)/(np.sqrt(6)+np.sqrt(3)+np.sqrt(2))
        w_cross = np.sqrt(3)/(np.sqrt(6)+np.sqrt(3)+np.sqrt(2))
        w_star = np.sqrt(2)/(np.sqrt(6)+np.sqrt(3)+np.sqrt(2))
        return((selem_plus, w_plus), (selem_cross, w_cross), (selem_star, w_star))
    else:
        return None

#%% VISUALIZATION FUNCTIONS

def black_plasma():
    colors = plt.cm.plasma(np.linspace(0,1,256))
    colors[:1, :] = np.array([0, 0, 0, 1])
    cmap = matplotlib.colors.ListedColormap(colors)
    return cmap

def white_viridis():
    colors = np.flip(plt.cm.viridis(np.linspace(0,1,256)),axis=0)
    colors[:1, :] = np.array([1, 1, 1, 1])
    cmap = matplotlib.colors.ListedColormap(colors)
    return cmap

def pl_black_plasma():
    c = black_plasma()(np.linspace(0, 1, 256))[:,0:3]
    pl_colorscale = []
    for i in range(256):
        pl_colorscale.append([i/255, f'rgb({c[i,0]},{c[i,1]},{c[i,2]})'])
    return pl_colorscale
 
def arrow_navigation(event, z, Z):
    if event.key == "up":
        z = min(z+1,Z-1)
    elif event.key == 'down':
        z = max(z-1,0)
    elif event.key == 'right':
        z = min(z+10,Z-1)
    elif event.key == 'left':
        z = max(z-10,0)
    elif event.key == 'pagedown':
        z = min(z+50,Z+1)
    elif event.key == 'pageup':
        z = max(z-50,0)
    return z

def show_vol(V, cmap=plt.cm.gray, vmin = None, vmax = None): 
    """
    Shows volumetric data for interactive inspection.
    @author: vand at dtu dot dk
    """
    def update_drawing():
        ax.images[0].set_array(V[z])
        ax.set_title(f'slice z={z}')
        fig.canvas.draw()
 
    def key_press(event):
        nonlocal z
        z = arrow_navigation(event,z,Z)
        update_drawing()
        
    Z = V.shape[0]
    z = (Z-1)//2
    fig, ax = plt.subplots()
    if vmin is None: 
        vmin = np.min(V)
    if vmax is None: 
        vmax = np.max(V)
    ax.imshow(V[z], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f'slice z={z}')
    fig.canvas.mpl_connect('key_press_event', key_press)

#%% HELPING FUNCTIONS

def create_test_volume(dim, sigma=7, threshold=0, boundary=0, frame = True, seed = None):
    """ Creates test volume for local thickness and porosity analysis.
    Arguments:
        dim: tuple giving the size of the volume
        sigma: smoothing scale, higher value - smoother objects
        threshold: a value close to 0, larger value - less material (smaller objects)
        boundary: strength of imposing object boundary pulled inwards
        frame: one-voxel frame of False 
    Returns:
        a test volume
    Example use:
        V = create_test_volume((150,100,50), boundary=0.1)
    For images (2D) use: 
        a = create_test_volume((50,50,1), frame=False)[:,:,0]
    Author: vand@dtu.dk, 2019
    """
    
    if len(dim)==3:
        r = np.fromfunction(lambda x, y, z: 
            ((x/(dim[0]-1)-0.5)**2+(y/(dim[1]-1)-0.5)**2+(z/(dim[2]-1)-0.5)**2)**0.5,
            dim, dtype=int)
    elif len(dim)==2:
        r = np.fromfunction(lambda x, y: 
            ((x/(dim[0]-1)-0.5)**2+(y/(dim[1]-1)-0.5)**2)**0.5, dim, dtype=int)
                 
    prng = np.random.RandomState(seed) # pseudo random number generator
    V = prng.standard_normal(dim)
    V[r>0.5] -= boundary;
    V = scipy.ndimage.gaussian_filter(V,sigma,mode='constant',cval=-boundary)
    V = V>threshold
    if frame:
        if len(dim)==3:
            V[[0,-1],:,:] = False
            V[:,[0,-1],:] = False
            V[:,:,[0,-1]] = False
        elif len (dim)==2:
            V[[0,-1],:] = False
            V[:,[0,-1]] = False
    return V

#%% VTK WRITE FUNCTIONS

def save_gray2vtk(volume, filename, filetype='ASCII', origin=(0,0,0),
                  spacing=(1,1,1), dataname='gray'):
    ''' Writes a vtk file with grayscace volume data.
    Arguments:
       volume: a grayscale volume, values will be saved as floats
       filename: filename with .vtk extension
       filetype: file type 'ASCII' or 'BINARY'. Writing a binary file might not
           work for all OS due to big/little endian issues.
       origin: volume origin, defaluls to (0,0,0)
       spacing: volume spacing, defaults to 1
       dataname: name associated with data (will be visible in Paraview)
    Author:vand@dtu.dk, 2019
    '''
    with open(filename, 'w') as f:
        # writing header
        f.write('# vtk DataFile Version 3.0\n')
        f.write('saved from python using save_gray2vtk\n')
        f.write('{}\n'.format(filetype))
        f.write('DATASET STRUCTURED_POINTS\n')
        f.write('DIMENSIONS {} {} {}\n'.format(\
                volume.shape[2],volume.shape[1],volume.shape[0]))
        f.write('ORIGIN {} {} {}\n'.format(origin[0],origin[1],origin[2]))
        f.write('SPACING {} {} {}\n'.format(spacing[0],spacing[1],spacing[2]))
        f.write('POINT_DATA {}\n'.format(volume.size))
        f.write('SCALARS {} float 1\n'.format(dataname))
        f.write('LOOKUP_TABLE default\n')
        
    # writing volume data
    if filetype.upper()=='BINARY':
        with open(filename, 'ab') as f:
            volume = volume.astype('float32') # Pareview expects 4-bytes float 
            volume.byteswap(True) # Paraview expects big-endian 
            volume.tofile(f)
    else: # ASCII
        with open(filename, 'a') as f:
            np.savetxt(f,volume.ravel(),fmt='%.5g', newline= ' ')
        
def save_rgba2vtk(rgba, dim, filename, filetype='ASCII'):
    ''' Writes a vtk file with RGBA volume data.
    Arguments:
       rgba: an array of shape (N,4) containing RGBA values
       dim: volume shape, such that prod(dim) = N
       filename: filename with .vtk extension
       filetype: file type 'ASCII' or 'BINARY'. Writing a binary file might not
           work for all OS due to big/little endian issues.
    Author:vand@dtu.dk, 2019
    '''    
    with open(filename, 'w') as f:
        # writing header
        f.write('# vtk DataFile Version 3.0\n')
        f.write('saved from python using save_rgba2vtk\n')
        f.write('{}\n'.format(filetype))
        f.write('DATASET STRUCTURED_POINTS\n')
        f.write('DIMENSIONS {} {} {}\n'.format(dim[2],dim[1],dim[0]))
        f.write('ORIGIN 0 0 0\n')
        f.write('SPACING 1 1 1\n')
        f.write('POINT_DATA {}\n'.format(np.prod(dim)))
        f.write('COLOR_SCALARS rgba 4\n')
    
    # writing color data
    if filetype.upper()=='BINARY':
        with open(filename, 'ab') as f:
            rgba = (255*rgba).astype('ubyte') # Pareview expects unsigned char  
            rgba.byteswap(True) # Paraview expects big-endian 
            rgba.tofile(f)
    else: # ASCII
        with open(filename, 'a') as f:
            np.savetxt(f,rgba.ravel(),fmt='%.5g', newline= ' ')   

def _tofootprint(selem):
    return (np.array(selem).astype(bool),1)

def save_thickness2vtk(B, thickness, filename, colormap = black_plasma(), 
                  maxval = None, selem_radius = 1, filetype='ASCII', origin=(0,0,0),
                  spacing=(1,1,1)):
    ''' Writes a vtk file with results of local thickness analysis.
    Author:vand@dtu.dk, 2019
    '''
    
    g = scipy.ndimage.distance_transform_edt(B) - scipy.ndimage.distance_transform_edt(~B) - B + 0.5
    g = np.exp(0.1*g)
    g = g/(g+1)
    
    save_gray2vtk(g, filename, filetype=filetype, origin=origin, spacing = spacing)
    
    if maxval is None: 
        maxval = np.max(thickness)
    
    if selem_radius is not None:
        thickness = skimage.morphology.dilation(thickness, 
                footprint=_tofootprint(skimage.morphology.ball(selem_radius)))
   
    rgba = colormap(thickness.ravel()/maxval)

    with open(filename, 'a') as f:
        f.write('COLOR_SCALARS rgba 4\n')
    
    # writing color data
    if filetype.upper()=='BINARY':
        with open(filename, 'ab') as f:
            rgba = (255*rgba).astype('ubyte') # Pareview expects unsigned char  
            rgba.byteswap(True) # Paraview expects big-endian 
            rgba.tofile(f)
    else: # ASCII
        with open(filename, 'a') as f:
            np.savetxt(f,rgba.ravel(),fmt='%.5g', newline= ' ')  