#%% import modules
import skimage as sk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import pandas as pd
import math
from scipy import ndimage as ndi
from scipy import stats
import imageio
import nd2
from processingFunctions import *
import warnings
warnings.filterwarnings(action='once')

#%% path to folder with the images
path="/Users/romina/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/Y1/internship_1/remote_memory/NisslIR_mCherry_PVB_fosG"
#retrieve all files from dataset folder
all_files=[]
for filename in os.listdir(path):
    if filename.endswith(".tif"):
        all_files.append(filename)
    if filename.endswith(".nd2"):
        all_files.append(filename)
print(all_files)

# Choose the parameters for the image
## We recommend keeping the general parameters as they are (in exception of the channel which will depend on your dataset)
## We recommend inspecting a subset of images and adjusting Fos parameters to fit your dataset 

#%% General parameters
channelFos=1 #channel where Fos staining is
axis_limit = 60
dist_thresh=25
axis_min=11
circ=0.98
axis_ratio= 0.48

#Fos parameters 
fos_thresh={
    "is_intensity_low":0, #use 0 if you have images with a high background, change to 1 if your images have low intensity and you wish to enhance contrast 
    "top_thresh":25, #this is gonna be the threshold for images with higher background, adjust it based on your settings
    "mid_thresh":9,#this is gonna be the threshold for images with medium background, adjust it based on your settings
    "low_thresh":30, #this is gonna be the threshold for images with lower background, adjust it based on your settings
    "high_int_thresh":92,
    "low_int_thresh":30
}

#%% get intensity cut off values for fos filtering function
fos_ints= intensitySaver(path, all_files, channelFos, fos_thresh.get("is_intensity_low")).getIntensityValues()
print (fos_ints)

#%%
#select one image to visualize if the thresholding is correct
test_image= "V_2_2006.nd2" #name of image file
data= path + "/"+ test_image #open an image here to examinate (note that you're gonna have to open a few to see how your images vary from each other)
fos_t, stacks=getImg(channelFos, data)
print(stacks) #will print the number of planes of the z-stack

## visualize how your images look after being thresholded with the previous parameters
#%% visually inspect the chosen image and how it is being thresholded, and then adjust your thresholding values accordingly 
for i in range(stacks):
    fos_cc=fos_t[i]
    filt= getThresh(fos_cc).threshFos(fos_thresh, fos_ints)
    show_labels(filt, fos_cc, circ, axis_min, axis_limit, axis_ratio)

#%%
# print the counts for Fos cells
blobs_fos= getCoords(fos_cc, stacks, circ, axis_ratio, axis_min, axis_limit).coordsFos(fos_thresh, fos_ints)
overlap=getOverlap(stacks, dist_thresh).overlap_coords(blobs_fos)
print(len(blobs_fos)-len(overlap))

#%%
#get values of each plane to assess where to adjust values
intInfo= pd.DataFrame(columns=["stack","classifier value", "25 p", "99 p", "threshold applied"])
for i in range(stacks):
    denoise=sk.restoration.denoise_wavelet(fos_t[i])
    blurred = sk.filters.gaussian(denoise, sigma=2.0)
    if fos_thresh.get("is_intensity_low") ==0:
        prepro=blurred
    else:
        prepro= sk.exposure.equalize_adapthist(blurred, kernel_size=127,clip_limit=0.01,  nbins=256)
    thresh=sk.filters.threshold_otsu(prepro)
    if np.percentile(prepro, 25)<= fos_ints.get("low_int"): 
        cat= "high_int_thresh"
    elif np.percentile(prepro,99)>= fos_ints.get("high_int"):
        cat= "low_int_thresh"
    else:
        if thresh/np.median(prepro)>=fos_ints.get("int_cutoff_up"):
            cat= "no thresh"
        else:
            if thresh/np.median(prepro)<=fos_ints.get("int_cutoff"): 
                if thresh/np.median(prepro)<=fos_ints.get("int_cutoff_down"): 
                    cat= "low_thresh"
                else:
                    cat= "top thresh"
            else:
                cat= "mid thresh"
    val_list= [i, thresh/np.median(prepro),np.percentile(prepro, 25), np.percentile(prepro, 99), cat]
    c_series = pd.Series(val_list, index = intInfo.columns)
    intInfo = intInfo.append(c_series, ignore_index=True)
print(intInfo)

## once you have decided on the thresholding values copy them on the main file and run the script to obtain the counts for the dataset
# %%
