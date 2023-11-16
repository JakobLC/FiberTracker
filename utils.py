import matplotlib.pyplot as plt
import numpy as np
import scipy
import cv2
import jlc
from jlc.voltools import arrow_navigation
from collections import OrderedDict

def gaussian_filter_no_border(input,sigma,**kwargs):
    kwargs['mode'] = 'constant'
    kwargs['cval'] = 0
    vol_g = scipy.ndimage.gaussian_filter(input,sigma,**kwargs)
    vol_norm = scipy.ndimage.gaussian_filter(np.ones_like(input), sigma, **kwargs)+1e-12*input.max()
    vol_g /= vol_norm
    return vol_g

def estimate_bias_coefs(vol,q=0.1):
    q_coefs = np.array([np.quantile(v,q) for v in vol])
    q_coefs = q_coefs/np.mean(q_coefs)
    base_mean = vol.mean((1,2))
    y = base_mean/q_coefs
    (a,b) = np.polyfit(np.arange(len(y)),y,1)
    y_ideal = a*np.arange(len(y))+b
    coefs = y_ideal/y/q_coefs
    return coefs

def estimate_bias(vol):
    bias = vol[0]
    bias = scipy.ndimage.gaussian_filter(bias, sigma=2)
    bias = scipy.ndimage.median_filter(bias, size=[50,50])
    bias = scipy.ndimage.gaussian_filter(bias, sigma=10)
    return bias[None]

def norm_quantile(vol,alpha,clip=True):
    [q1,q2] = np.quantile(vol, [alpha, 1-alpha])
    if clip:
        vol = np.clip(vol, q1, q2)
    vol = (vol - q1) / (q2 - q1)
    return vol

def norm_translation(vol,
                     translate_search = np.arange(-8,8+1),
                     slice_x = slice(None),
                     slice_y = slice(None),
                     sigma = 0,
                     upscale = 10,
                     ref_index = 0.5,
                     bfts = np.arange(-24,24+1,2)):

    if isinstance(ref_index, int):
        frame1 = vol[ref_index, slice_x, slice_y]
    else:
        assert isinstance(ref_index, float)
        frame1 = vol[int(vol.shape[0]*ref_index),slice_x,slice_y]
    max_trans = np.max(np.abs(translate_search))
    d = translate_search[1] - translate_search[0]
    translate_search_big = np.linspace(translate_search[0] -d/2,
                                    translate_search[-1]+d/2,upscale*len(translate_search))
    d2 = bfts[1] - bfts[0]
    bf_trans_search_big = np.linspace(bfts[0] -d2/2,
                                    bfts[-1]+d2/2,upscale*len(bfts))
    if sigma>0:
        frame1 = scipy.ndimage.gaussian_filter(frame1, sigma)
    best_translation = []
    for f_i in range(vol.shape[0]):
        frame2 = vol[f_i, slice_x, slice_y]
        corr_mat = get_corr_mat(frame1,frame2,translate_search,sigma=0)
        corr_mat_big = cv2.resize(corr_mat, (0,0), fx=upscale, fy=upscale, interpolation=cv2.INTER_LANCZOS4)        
        max_idx = np.unravel_index(np.argmax(corr_mat_big), corr_mat_big.shape)
        best_translation.append((translate_search_big[max_idx[0]],translate_search_big[max_idx[1]]))
        bad_frame = np.abs(best_translation[-1][0])>=max_trans or np.abs(best_translation[-1][1])>=max_trans
        if bad_frame:
            print(f"bad frame {f_i}, using bf")
            corr_mat = get_corr_mat(frame1,frame2,bfts,sigma=0)
            corr_mat_big = cv2.resize(corr_mat, (0,0), fx=upscale, fy=upscale, interpolation=cv2.INTER_LANCZOS4)        
            max_idx = np.unravel_index(np.argmax(corr_mat_big), corr_mat_big.shape)
            best_translation[-1] = (bf_trans_search_big[max_idx[0]],bf_trans_search_big[max_idx[1]])
        if f_i==10:
            assert 1<0
        if f_i%50==0:
            print(f"done with frame {f_i}/{vol.shape[0]-1}")
    translated_vol = np.zeros_like(vol)
    for f_i in range(len(vol)):
        translation = np.float32([[1,0,best_translation[f_i][0]],[0,1,best_translation[f_i][1]]])
        translated_vol[f_i] = cv2.warpAffine(vol[f_i], translation,
                                                (vol[f_i].shape[1], vol[f_i].shape[0]), 
                                                flags=cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_REPLICATE)
    return translated_vol, np.array(best_translation)

def get_corr_mat(frame1,frame2,translate_search,sigma=0,zerozero_is_neighbours=True):
    corr_mat = np.zeros((len(translate_search), len(translate_search)))
    for x in translate_search:
        for y in translate_search:  
            translation = np.float32([[1,0,x],[0,1,y]])
            frame2_translated = cv2.warpAffine(frame2, translation, (frame2.shape[1], frame2.shape[0]), flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            if sigma>0:
                frame2_translated_g = scipy.ndimage.gaussian_filter(frame2_translated, sigma)
            else:
                frame2_translated_g = frame2_translated
            corr = -np.mean((frame1-frame2_translated_g)**2)
            corr_mat[translate_search==x, translate_search==y] = corr
    if zerozero_is_neighbours:
        i =  np.where(translate_search==0)[0][0]
        zero = [corr_mat[i,i+1],
                corr_mat[i,i-1],
                corr_mat[i+1,i],
                corr_mat[i-1,i]]
        corr_mat[i,i] = sum(zero)/len(zero)
    return corr_mat
    
def process_vol(filename):
    vol = jlc.load_tifvol(filename)
    vol = vol[:,:,:vol.shape[2]//2].astype(float)
    vol = norm_quantile(vol,alpha=0.001,clip=True)
    coefs = estimate_bias_coefs(vol)
    vol *= coefs.reshape(-1,1,1)
    bias = estimate_bias(vol)
    vol -= bias
    vol = norm_quantile(vol,alpha=0.001,clip=True)
    vol,best_translation = norm_translation(vol)
    return vol

def soft_threshold(vol,start=0,stop=1,power=1): 
    to_interval = lambda x: np.clip((x-start)/(stop-start),0,1)
    f = lambda x: h(to_interval(x),power)
    return f(vol)
def h(x,p):
    return (x<0.5)*0.5*2**p*x**p+(x>=0.5)*(1-0.5*2**p*(1-x)**p)

def get_disc_2d(r,binarize=True):
    k = r
    kf = np.floor(k).astype(int)
    X,Y = np.meshgrid(*[np.arange(-kf,kf+1)]*2)
    out = X**2+Y**2-k**2<=0 if binarize else np.sqrt(X**2+Y**2)-k
    return out

class FiberGraph:
    def __init__(self, find_conn_func=None, binarize_func=None, image_size=None, radius=3, threshold=0.15):
        self.nodes = []
        self.edges = []
        self.cc_to_nodes = []

        self.image_size = image_size
        self.XY = np.zeros((0,2))
        self.node_image = None
        self.radius = radius
        self.debug = False
        self.threshold = threshold
        
        if binarize_func is None:
            self.binarize_func = self.default_binarize_func
        else:
            self.binarize_func = binarize_func

        if find_conn_func is None:
            self.find_conn_func = self.default_find_conn_func
        else:
            self.find_conn_func = find_conn_func
        

    def add_node(self,x,y,timestep,conn=None,growing=None):
        xy_inbounds = 0<=int(np.round(x))<self.image_size[1] and 0<=int(np.round(y))<self.image_size[0]
        assert xy_inbounds, f"node is out of bounds. image size={self.image_size}, x={x}, y={y}"
        if self.node_image is None and self.image_size is not None:
            self.node_image = np.zeros(self.image_size)
        if self.node_image is not None:
            self.node_image[int(np.round(y)),int(np.round(x))] = 1
        node = {"idx": len(self.nodes),
                "x": x,
                "y": y,
                "active": True,
                "active_t": [timestep,int(1e10)],
                "t": timestep,
                "growing": True if growing is None else growing,
                "bad_counter": 0,
                "cc_idx": None,
                "old_versions": {}}
        
        if conn is None and self.num_fibers()==0:
            conn = []
        if conn is None:
            conn = self.find_conn_func(self.XY,self.nodes,node,self.radius)
        if isinstance(conn,int):
            if isinstance(conn,int):
                conn = [conn]
        assert isinstance(conn,list)
        
        for c in conn:
            self.nodes[c]["conn"].append(node["idx"])
            self.nodes[c]["growing"] = False

        node["conn"] = conn
        if len(conn)>0:
            node["root"] = False
            cc_idx = self.nodes[conn[0]]["cc_idx"]
            self.cc_to_nodes[cc_idx].append(node["idx"])
        else:
            node["root"] = True
            cc_idx = len(self.cc_to_nodes)
            
            self.cc_to_nodes.append([node["idx"]])
        node["cc_idx"] = cc_idx
        self.nodes.append(node)
        self.XY = np.vstack((self.XY,[x,y]))

    def __len__(self):
        return len(self.nodes)

    def num_fibers(self):
        return len(self.cc_to_nodes)
    
    def plot(self,mpl_colors=True,color=None,plot_cc=None,t=None):
        if t is None:
            t = np.max([n["t"] for n in self.nodes])
        t_mask = [n["active_t"][0]<=t<=n["active_t"][1] for n in self.nodes]
        if plot_cc is None:
            plot_cc = np.arange(self.num_fibers())
        if mpl_colors:
            color = [f"C{i%10}" for i in range(self.num_fibers())]
        else:
            if color is None:
                color = np.random.rand(self.num_fibers(),3)
            else:
                if isinstance(color,str):
                    color = np.array([color]*self.num_fibers())
                elif isinstance(color,(list,np.ndarray)):
                    assert len(color)==self.num_fibers(),"color list must have same length as number of fibers if color is array/list"
        for cc_idx in range(len(self.cc_to_nodes)):
            if cc_idx in plot_cc:
                edges = []
                for node_idx in self.cc_to_nodes[cc_idx]:
                    if t_mask[node_idx]:
                        for c in self.nodes[node_idx]["conn"]:
                            if t_mask[c]:
                                edges.append((node_idx,c))
                edges_x1_x2_nan = []
                edges_y1_y2_nan = []
                for idx1,idx2 in edges:
                    x1,y1 = self.nodes[idx1]["x"],self.nodes[idx1]["y"]
                    x2,y2 = self.nodes[idx2]["x"],self.nodes[idx2]["y"]
                    edges_x1_x2_nan += [x1,x2,np.nan]
                    edges_y1_y2_nan += [y1,y2,np.nan]
                plt.plot(edges_x1_x2_nan,edges_y1_y2_nan,".-",color=color[cc_idx])
        XY_recent = self.XY[np.logical_and([n["t"]+10>=t for n in self.nodes],t_mask)]
        plt.plot(XY_recent[:,0],XY_recent[:,1],"o",color="green",markerfacecolor='none',linewidth=1)
        XY_root = self.XY[np.logical_and([n["root"] for n in self.nodes],t_mask)]
        plt.plot(XY_root[:,0],XY_root[:,1],"o",color="red",markerfacecolor='none',markersize=10)

    def default_find_conn_func(self,xy,nodes,node,r):
        dist = np.linalg.norm(xy-np.array([node["x"],node["y"]]),axis=1)
        t = np.array([n["t"] for n in nodes if n["active"]])
        t_now = node["t"]
        mod_dist = dist#-5.0*self.radius*(1-soft_threshold(t_now-t-1,0,10,3))
        active_idx = np.array([n["idx"] for n in nodes if n["active"]])
        mod_dist = mod_dist[active_idx]
        if any(mod_dist<r*3+2):
            return [active_idx[np.argmin(mod_dist)]]
        else:
            return []
        
    def default_binarize_func(self,frame):
        return scipy.ndimage.binary_opening(frame>self.threshold,structure=np.ones((2,2)),iterations=1)
    
    def mask_out_func(self,radius=None):
        if radius is None:
            radius = self.radius
        assert self.node_image is not None
        if len(self.nodes)==0:
            return 1
        mask = self.node_image.copy()
        mask = soft_threshold(scipy.ndimage.distance_transform_edt(mask==0)-radius-0.5)
        return mask

    def process_frame(self,frame,t):
        if len(self.nodes)>0:
            Y = np.round(self.XY[:,1]).astype(int)
            X = np.round(self.XY[:,0]).astype(int)
            node_is_bad = np.logical_not(self.binarize_func(frame)[Y,X])
            nodes_for_removal = []
            for i in range(len(self.nodes)):
                n = self.nodes[i]
                if node_is_bad[i]:
                    self.nodes[i]["bad_counter"] += 1
                    if n["bad_counter"]==3 and len(n["conn"])<=1:
                        nodes_for_removal.append(i)
                else:
                    n["bad_counter"] = 0
            if len(nodes_for_removal)<10:
                for i in nodes_for_removal:
                    self.nodes[i]["active"] = False
                    self.nodes[i]["active_t"][1] = t
            else:
                for i in nodes_for_removal:
                    self.nodes[i]["bad_counter"] -= 1
        mask_out = self.mask_out_func()
        BW = self.binarize_func(frame*mask_out)
        maxima = strict_local_max(frame*BW)
        Y,X = np.where(maxima)
        m_vals = frame[Y,X]
        m_order = np.argsort(m_vals)[::-1]
        X,Y,m_vals = X[m_order],Y[m_order],m_vals[m_order]
        for i in range(len(m_order)):
            if i==0:
                self.add_node(X[i],Y[i],t)
            elif np.min((X[i]-X[:i])**2+(Y[i]-Y[:i])**2)>=self.radius**2:
                self.add_node(X[i],Y[i],t)
        
    def process_vol(self,vol,max_frames=100000):
        if self.image_size is not None:
            assert vol.shape[1:]==self.image_size
        else:
            self.image_size = vol.shape[1:]
        if self.node_image is None:
            self.node_image = np.zeros(self.image_size)
            for x,y in self.XY:
                self.node_image[int(np.round(y)),int(np.round(x))] = 1
        num_frames = min(vol.shape[0],max_frames)
        for t in range(num_frames):
            self.process_frame(vol[t],t)

def local_max(frame):
    return scipy.ndimage.maximum_filter(frame,size=(3,3)) == frame

def strict_local_max(frame):
    fp = np.ones((3,3))
    fp[1,1] = 0
    return scipy.ndimage.maximum_filter(frame,footprint=fp) < frame

def inspect_fiber_vol(V, cmap=plt.cm.gray, vmin = None, vmax = None, fiber_graph=None, fiber_plot_kwargs={}):
    def update_drawing():
        ax.images[0].set_array(V[z])
        if fiber_graph is not None:
            for line in ax.lines:
                line.remove()
            fiber_graph.plot(t=z,**fiber_plot_kwargs)
        ax.set_title(f'slice z={z}/{Z}')
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
    fiber_graph.plot(t=z,**fiber_plot_kwargs)
    ax.set_title(f'slice z={z}/{Z}')
    fig.canvas.mpl_connect('key_press_event', key_press)