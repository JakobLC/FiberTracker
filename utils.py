import matplotlib.pyplot as plt
import numpy as np
import scipy
import cv2
import jlc
from jlc.voltools import arrow_navigation
from collections import OrderedDict
from mpl_toolkits.mplot3d import axes3d


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
    if isinstance(alpha,list):
        [q1,q2] = np.quantile(vol, [alpha[0], 1-alpha[1]])
    else:
        assert isinstance(alpha, float)
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
                     bfts = np.arange(-24,24+1,2),
                     reset_origo_on_bad_frame=False):
    origo = [0,0]
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
        corr_mat = get_corr_mat(frame1,frame2,translate_search,sigma=0,origo=origo)
        corr_mat_big = cv2.resize(corr_mat, (0,0), fx=upscale, fy=upscale, interpolation=cv2.INTER_LANCZOS4)        
        max_idx = np.unravel_index(np.argmax(corr_mat_big), corr_mat_big.shape)
        best_translation.append((translate_search_big[max_idx[0]]+origo[0],
                                 translate_search_big[max_idx[1]]+origo[1]))
        bad_frame = np.abs(best_translation[-1][0])>=max_trans or np.abs(best_translation[-1][1])>=max_trans
        if bad_frame:
            print(f"bad frame {f_i}, using bf")
            corr_mat = get_corr_mat(frame1,frame2,bfts,sigma=0,origo=origo)
            corr_mat_big = cv2.resize(corr_mat, (0,0), fx=upscale, fy=upscale, interpolation=cv2.INTER_LANCZOS4)        
            max_idx = np.unravel_index(np.argmax(corr_mat_big), corr_mat_big.shape)
            best_translation[-1] = (bf_trans_search_big[max_idx[0]]+origo[0],
                                    bf_trans_search_big[max_idx[1]]+origo[1])
            if reset_origo_on_bad_frame:
                origo = int(round(best_translation[-1][0])),int(round(best_translation[-1][1]))
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

def get_corr_mat(frame1,frame2,translate_search,sigma=0,zerozero_is_neighbours=True,origo=[0,0]):
    corr_mat = np.zeros((len(translate_search), len(translate_search)))
    for x in translate_search+origo[0]:
        for y in translate_search+origo[1]:  
            translation = np.float32([[1,0,x],[0,1,y]])
            frame2_translated = cv2.warpAffine(frame2, translation, (frame2.shape[1], frame2.shape[0]), flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            if sigma>0:
                frame2_translated_g = scipy.ndimage.gaussian_filter(frame2_translated, sigma)
            else:
                frame2_translated_g = frame2_translated
            corr = -np.mean((frame1-frame2_translated_g)**2)
            corr_mat[translate_search==x, translate_search==y] = corr
    if zerozero_is_neighbours and (0 in (translate_search+origo[0]).tolist() and 0 in (translate_search+origo[1]).tolist()):
        i1 =  np.where(translate_search+origo[0]==0)[0][0]
        i2 =  np.where(translate_search+origo[1]==0)[0][0]
        zero = [corr_mat[i1,i2+1],
                corr_mat[i1,i2-1],
                corr_mat[i1+1,i2],
                corr_mat[i1-1,i2]]
        corr_mat[i1,i2] = sum(zero)/len(zero)
    return corr_mat
    
def process_vol(filename,smaller=False,reset_origo_on_bad_frame=False):
    vol = jlc.load_tifvol(filename)
    vol = vol[:,:,:vol.shape[2]//2].astype(float)
    if smaller:
        vol = vol[:,vol.shape[0]//4:vol.shape[0]//4*3,vol.shape[1]//4:vol.shape[1]//4*3]
    vol = norm_quantile(vol,alpha=0.001,clip=True)
    coefs = estimate_bias_coefs(vol)
    vol *= coefs.reshape(-1,1,1)
    bias = estimate_bias(vol)
    vol -= bias
    vol = norm_quantile(vol,alpha=0.001,clip=True)
    vol,best_translation = norm_translation(vol,reset_origo_on_bad_frame=reset_origo_on_bad_frame)
    return vol,best_translation

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
        self.XY = np.zeros((0,2))

        self.high_dens_ratio = 0
        self.image_size = image_size
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
                "cc_idx": None}
        
        if conn is None and self.num_fibers()==0:
            conn = []
        if conn is None:
            conn = self.find_conn_func(self.XY,self.nodes,node)
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
    
    def get_colors(self,mpl_colors=True,color=None,plot_cc=None):
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
        return color,plot_cc
    
    def plot_3d(self,mpl_colors=True,color=None,plot_cc=None,image=None):
        t_mask = [n["active"] for n in self.nodes]
        color,plot_cc = self.get_colors(mpl_colors=mpl_colors,color=color,plot_cc=plot_cc)
        ax = plt.figure().add_subplot(projection='3d')
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
                edges_t1_t2_nan = []
                for idx1,idx2 in edges:
                    x1,y1 = self.nodes[idx1]["x"],self.nodes[idx1]["y"]
                    x2,y2 = self.nodes[idx2]["x"],self.nodes[idx2]["y"]
                    t1,t2 = self.nodes[idx1]["t"],self.nodes[idx2]["t"]
                    edges_x1_x2_nan += [x1,x2,np.nan]
                    edges_y1_y2_nan += [y1,y2,np.nan]
                    edges_t1_t2_nan += [t1,t2,np.nan]
                ax.plot(edges_x1_x2_nan,edges_y1_y2_nan,edges_t1_t2_nan,".-",color=color[cc_idx])
        
        ax.set_xlim3d(0,self.image_size[1])
        ax.set_ylim3d(0,self.image_size[0])
        ax.set_zlim3d(0,np.max([n["t"] for n in self.nodes]))
        
        if image is not None:
            #show in the t=0 plane
            ax.imshow(image,extent=[0,image.shape[1],image.shape[0],0],alpha=0.5)

    def plot(self,mpl_colors=True,color=None,plot_cc=None,t=None):
        if t is None:
            t = np.max([n["t"] for n in self.nodes])
        t_mask = [n["active_t"][0]<=t<=n["active_t"][1] for n in self.nodes]
        color,plot_cc = self.get_colors(mpl_colors=mpl_colors,color=color,plot_cc=plot_cc)

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

    def default_find_conn_func(self,xy,nodes,node,growing_reward=3):
        dist = np.linalg.norm(xy-np.array([node["x"],node["y"]]),axis=1)
        mod_dist = dist
        active_idx = np.array([n["idx"] for n in nodes if n["active"]])
        mod_dist = mod_dist[active_idx]
        growing_idx = np.array([k for (k,i) in enumerate(active_idx) if nodes[i]["growing"]])
        mod_dist[growing_idx] -= growing_reward
        if any(mod_dist<8):
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
        self.max_t = t
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

    def leaf_crop(self,num=2):
        #loops over all leaf nodes (len(conn)==1) and removes them
        #if they are are within num nodes of a node with len(conn)>2 (a branching node)
        for i,n in enumerate(self.nodes):
            jj = [i]
            if len(n["conn"])==1:
                c = n["conn"][0]
                for _ in range(num):
                    if not self.nodes[c]["active"]:
                        break
                    active_conn = [c for c in self.nodes[c]["conn"] if self.nodes[c]["active"]]
                    if len(active_conn)<=1:
                        break
                    elif len(active_conn)>2:
                        for j in jj:
                            self.nodes[j]["active"] = False
                            self.nodes[j]["active_t"][1] = self.max_t
                        break
                    else:
                        jj.append(c)
                        c = active_conn[0]
                        
    def cc_crop(self,min_size=3):
        #loops over all connected components and removes them if they are smaller than min_size
        for i,cc in enumerate(self.cc_to_nodes):
            if len(cc)<min_size:
                for j in cc:
                    self.nodes[j]["active"] = False
                    self.nodes[j]["active_t"][1] = self.max_t
    
    def high_dens_crop(self,high_dens_image):
        for i,n in enumerate(self.nodes):
            if high_dens_image[int(np.round(n["y"])),int(np.round(n["x"]))]:
                self.nodes[i]["active"] = False
                self.nodes[i]["active_t"][1] = self.max_t
        self.high_dens_ratio = np.sum(high_dens_image)/high_dens_image.size
    
    def remove_inactive(self):
        #removes all inactive nodes and updates the self.cc_to_nodes list as well as self.XY and self.nodes and self.edges
        #and adjusts all index
        old_active = [n["active"] for n in self.nodes]
        self.XY = self.XY[old_active]
        old_to_new = {i:j for j,i in enumerate(np.where(old_active)[0])}
        self.nodes = [n for n in self.nodes if n["active"]]
        
        self.edges = []
        for i in range(len(self.nodes)):
            self.nodes[i]["conn"] = [old_to_new[c] for c in self.nodes[i]["conn"] if old_active[c]]
            self.nodes[i]["idx"] = i
            self.nodes[i]["cc_idx"] = None
            self.edges += [(i,c) for c in self.nodes[i]["conn"]]
        self.cc_to_nodes = []
        for _ in range(len(self.nodes)):
            has_cc = [n["cc_idx"] is not None for n in self.nodes]
            if all(has_cc):
                break
            first_node = has_cc.index(False)
            cc_node_idx = self.get_cc_node_idx(first_node)
            for c in cc_node_idx:
                self.nodes[c]["cc_idx"] = len(self.cc_to_nodes)
            self.cc_to_nodes.append(cc_node_idx)

    def get_cc_node_idx(self,node_idx):
        upper_limit_recursive=len(self.nodes)
        cc_node_idx = [node_idx]
        neighbours = self.nodes[node_idx]["conn"]
        for _ in range(upper_limit_recursive):
            new_neighbours = [c for c in neighbours if c not in cc_node_idx]
            if len(new_neighbours)==0:
                break
            cc_node_idx += new_neighbours
            neighbours = sum([self.nodes[c]["conn"] for c in new_neighbours],[])
        return cc_node_idx 
    
    def find_long_seperated_paths(self,max_dist=5,min_len=10):
        get_other = lambda x,check_x: x[0] if x[1]==check_x else x[1]
        dist_to_closest_cc = np.zeros(len(self.nodes))
        for i in range(len(self.nodes)):
            cc = self.cc_to_nodes[self.nodes[i]["cc_idx"]]
            not_cc_XY = self.XY[[j for j in range(len(self.nodes)) if j not in cc]]
            dist_to_closest_cc[i] = np.min(np.linalg.norm(not_cc_XY-self.XY[i],axis=1))
        paths = []
        paths_cc = []
        for cc_idx,cc_nodes_idx in enumerate(self.cc_to_nodes):
            if len(cc_nodes_idx)<min_len:
                continue
            #find set of paths where each node has len(conn)==2 contained in the cc
            
            used_idx = [False for _ in range(len(cc_nodes_idx))]
            conns = [self.nodes[i]["conn"] for i in cc_nodes_idx]
            to_new_idx = {i:j for j,i in enumerate(cc_nodes_idx)}
            to_old_idx = {j:i for j,i in enumerate(cc_nodes_idx)}
            conns = [[to_new_idx[c] for c in conn] for conn in conns]
            for _ in range(len(cc_nodes_idx)):
                len_2_conn = [i for i in range(len(cc_nodes_idx)) if len(conns[i])==2 and not used_idx[i]]
                if len(len_2_conn)==0:
                    break
                path = [len_2_conn[0]]
                used_idx[len_2_conn[0]] = True
                #continue in each direction until either len(conn)==3 or len(conn)==1
                next_node = conns[path[0]][1]
                for _ in range(len(cc_nodes_idx)):
                    if not len(conns[next_node])==2:
                        break
                    path.append(next_node)
                    used_idx[next_node] = True
                    next_node = get_other(conns[next_node],next_node)
                prev_node = conns[path[0]][0]
                for _ in range(len(cc_nodes_idx)):
                    if not len(conns[prev_node])==2:
                        break
                    path.insert(0,prev_node)
                    used_idx[prev_node] = True
                    prev_node = get_other(conns[prev_node],prev_node)
                path = [to_old_idx[i] for i in path]
                if len(path)>=min_len:
                    paths.append(path)
                    paths_cc.append(cc_idx)
        n = len(paths)
        print("found n="+str(n)+" candidate paths with len>="+str(min_len))
        path_mean_min_dist = np.zeros(n)
        for i,path in enumerate(paths):
            path_mean_min_dist[i] = np.mean(dist_to_closest_cc[path])
        mask = path_mean_min_dist>max_dist
        print("found n="+str(sum(mask))+" candidate paths with mean min dist>"+str(max_dist))
        paths = [paths[i] for i in range(n) if mask[i]]
        paths_cc = [paths_cc[i] for i in range(n) if mask[i]]
        path_mean_min_dist = [path_mean_min_dist[i] for i in range(n) if mask[i]]
        dist = distance_transform_upscale(self.image_size,paths,self.XY)
        return paths,paths_cc,path_mean_min_dist,dist
    
    def estimate_stats(self,image,image_segment=None,high_dens_image=None,max_dist=5,min_len=10):
        paths,paths_cc,path_mean_min_dist,dist = self.find_long_seperated_paths(max_dist=max_dist,min_len=min_len)
        if len(paths)==0:
            raise Exception("no paths found")
        len_of_long_paths = 0
        growth_rate = []
        for path in paths:
            path_XY = self.XY[path]
            path_time = np.array([abs(self.nodes[path[i]]["t"]-self.nodes[path[i+1]]["t"]) for i in range(len(path)-1)])
            path_dist = np.linalg.norm(path_XY[1:]-path_XY[:-1],axis=1)
            growth_rate.append((path_dist,path_time))
            len_of_long_paths += np.sum(path_dist)
        avg_growth_rate = 0#np.mean(sum([v.tolist() for v in growth_rate],[]))

        intensity_of_long_paths = image[dist<1.5*self.radius].sum()
        intensity_per_length = intensity_of_long_paths/len_of_long_paths

        total_intensity = image.sum()
        total_length_intensity = total_intensity/intensity_per_length

        if high_dens_image is None:
            assert image_segment is not None
            high_dens_image = get_high_density_mask(image_segment)
        low_intensity = image[high_dens_image==0].sum()
        low_length_intensity = low_intensity/intensity_per_length 
        high_dens_ratio = np.mean(high_dens_image)

        low_length_graph = 0
        for edge in self.edges:
            low_length_graph += np.linalg.norm(self.XY[edge[0]]-self.XY[edge[1]])

        stats = {"intensity_per_length": intensity_per_length,
                 "intensity_of_long_paths": intensity_of_long_paths,
                 "total_intensity": total_intensity,
                 "low_intensity": low_intensity,
                 "len_of_long_paths": len_of_long_paths,
                 "total_length_intensity": total_length_intensity,
                 "low_length_graph": low_length_graph,
                 "low_length_intensity": low_length_intensity,
                 "avg_growth_rate": avg_growth_rate,
                 "high_dens_ratio": high_dens_ratio}
        return stats, growth_rate, paths
    
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


def laplace(x,s1=0.75,s2=1.5,clip0=True):
    x = scipy.ndimage.gaussian_filter(x, sigma=(0, s1, s1))-scipy.ndimage.gaussian_filter(x, sigma=(0, s2, s2))
    if clip0:
        x[x<0] = 0
    return x

def filter_vol(vol,s1 = 0.75, s2 = 1.5, strict_max_filter=True):
    vol = scipy.ndimage.median_filter(vol, size=(9,1,1), mode='mirror')
    vol = np.maximum(norm_quantile(laplace(vol,s1,s2),0.01)*soft_threshold(vol,0.08,0.12),soft_threshold(vol,0.5,0.75))
    if strict_max_filter:
        kernel = (10**np.linspace(-4,-2,3)).tolist()
        kernel = np.array(kernel+[1-2*sum(kernel)]+kernel[::-1])
        #filter in x and y
        vol = scipy.ndimage.convolve1d(vol, kernel, axis=1, mode='mirror')
        vol = scipy.ndimage.convolve1d(vol, kernel, axis=2, mode='mirror')
    return vol

def find_conn2(xy,nodes,node,r,upper_limit=15,at_most_prev_nodes=10,at_most_prev_t=100,good_speed=0.15):
    dist = np.linalg.norm(xy-np.array([node["x"],node["y"]]),axis=1)
    dist_idx = np.where(dist<=upper_limit)[0]
    dist_idx = np.array([i for i in dist_idx if nodes[i]["active"]])
    if len(dist_idx)==0:
        return []
    #split into growing and non-growing:
    growing_index = np.array([i for i in dist_idx if nodes[i]["growing"]])
    cost = []
    t_now = node["t"]
    for i in growing_index:
        path = [nodes[i]]
        path_idx = [i]
        for _ in range(at_most_prev_nodes):
            if len(path[-1]["conn"])==0:
                break
            j = path[-1]["conn"][0]
            if nodes[j]["root"]:
                break
            path.append(nodes[j])
            path_idx.append(j)
            if path[-1]["t"]<t_now-at_most_prev_t:
                break
        if not len(path)==len(np.unique(path_idx)):
            print(path_idx)
            for j in np.unique(path_idx):
                print("NODE "+str(j))
                print(nodes[j])
            raise Exception("path_idx not unique")
        if len(path)>1:
            xy_path = np.array([[n["x"],n["y"]] for n in path])
            cum_dist = np.linalg.norm(xy_path[1:]-xy_path[:-1],axis=1).sum()
            cum_t = t_now-path[-1]["t"]
            speed = cum_dist/cum_t
            gs = good_speed
            speed = 1-soft_threshold(speed,0.5*gs,1.5*gs)
            speed *= 1-soft_threshold(len(path),0,3)
            v1 = np.array([node["x"],node["y"]])-xy_path[0]
            v2 = xy_path[1]-xy_path[0]
            angle = np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
            angle = 1-angle/np.pi
            angle = 1-soft_threshold(angle,0.6,0.9,2)
        else:
            angle = 0
            speed = 0
        #cost formula
        cost.append(dist[i]*(1+1.0*speed)+1.0*angle)
    if node["x"]==104 and node["y"]==36:
        print(cost)
    if any([c<r*3+4 for c in cost]):
        return [growing_index[np.argmin(cost)]]
    else:
        non_growing_index = np.array([i for i in dist_idx if not nodes[i]["growing"]])
        if len(non_growing_index)==0:
            return []
        cost = dist[non_growing_index]
        if any(cost<r*3+2):
            return [non_growing_index[np.argmin(cost)]]
        else:
            return []
        
def get_high_density_mask(im,thresh=0.4,sigma=5,min_cc=100):
    mask = scipy.ndimage.gaussian_filter(im,sigma=5)>0.4
    #remove binary cc with less than min_cc pixels
    cc = scipy.ndimage.label(mask)[0]
    for i in range(1,cc.max()+1):
        if (cc==i).sum()<min_cc:
            mask[cc==i] = False
    return mask

def distance_transform_upscale(imsize,paths,XY,upscale=4):
    dist_big = np.ones((imsize[0]*upscale,imsize[1]*upscale))
    for path in paths:
        path_XY = XY[path]
        max_segment_len = np.max(np.linalg.norm(path_XY[1:]-path_XY[:-1],axis=1))
        n_linspace = int(np.ceil(max_segment_len*upscale*2))
        x_upscale_list = np.array(sum([np.linspace(path_XY[i,0],path_XY[i+1,0],n_linspace).tolist() for i in range(len(path_XY)-1)],[]))
        y_upscale_list = np.array(sum([np.linspace(path_XY[i,1],path_XY[i+1,1],n_linspace).tolist() for i in range(len(path_XY)-1)],[]))
        x_upscale_list = np.round(x_upscale_list*upscale).astype(int)
        y_upscale_list = np.round(y_upscale_list*upscale).astype(int)
        dist_big[y_upscale_list,x_upscale_list] = 0
    dist_big = scipy.ndimage.distance_transform_edt(dist_big)/upscale
    dist = cv2.resize(dist_big,imsize[::-1],interpolation=cv2.INTER_AREA)
    return dist