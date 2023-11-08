import matplotlib.pyplot as plt
import numpy as np
import scipy

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
    return bias        

def norm_quantile(vol,alpha,clip=True):
    [q1,q2] = np.quantile(vol, [alpha, 1-alpha])
    if clip:
        vol = np.clip(vol, q1, q2)
    vol = (vol - q1) / (q2 - q1)
    return vol