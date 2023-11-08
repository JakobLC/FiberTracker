import matplotlib.pyplot as plt
import numpy as np
import scipy
import cv2
import jlc

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
                     slice_x = slice(100,200),
                     slice_y = slice(100,200),
                     sigma = 0,
                     upscale = 10,
                     ref_index = 0.5):
    
    if isinstance(ref_index, int):
        frame1 = vol[ref_index, slice_x, slice_y]
    else:
        assert isinstance(ref_index, float)
        frame1 = vol[int(vol.shape[0]*ref_index),slice_x,slice_y]
    d = translate_search[1] - translate_search[0]
    translate_search_big = np.linspace(translate_search[0] -d/2,
                                    translate_search[-1]+d/2,upscale*len(translate_search))
    corr_mat = np.zeros((len(translate_search), len(translate_search)))
    if sigma>0:
        frame1 = scipy.ndimage.gaussian_filter(frame1, sigma)
    best_translation = []
    for f_i in range(vol.shape[0]):
        for x in translate_search:
            for y in translate_search:  
                frame2 = vol[f_i, slice_x, slice_y]
                translation = np.float32([[1,0,x],[0,1,y]])
                frame2_translated = cv2.warpAffine(frame2, translation, (frame2.shape[1], frame2.shape[0]), flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                if sigma>0:
                    frame2_translated_g = scipy.ndimage.gaussian_filter(frame2_translated, sigma)
                else:
                    frame2_translated_g = frame2_translated
                corr = -np.mean((frame1-frame2_translated_g)**2)
                corr_mat[translate_search==x, translate_search==y] = corr
        corr_mat_big = cv2.resize(corr_mat, (0,0), fx=upscale, fy=upscale, interpolation=cv2.INTER_LANCZOS4)        
        max_idx = np.unravel_index(np.argmax(corr_mat_big), corr_mat_big.shape)
        best_translation.append((translate_search_big[max_idx[0]],translate_search_big[max_idx[1]]))
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