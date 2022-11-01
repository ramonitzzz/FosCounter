#%% <import modules>
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
warnings.filterwarnings('ignore')

#%% PARAMETERS
# you're gonna have to adjust the thresholding parameters depending on your images 
channel=1 #channel where Fos staining is
is_intensity_low=1 #change to 1 if your images have low intensity and you wish to enhance contrast
axis_limit = 60
dist_parameter=25
axis_min=11
circ=0.98
axis_ratio= 0.48

#Fos parameters, if you want to filter more per image, then 
fos_thresh={
    "is_intensity_low":0, #use 0 if you have images with a high background, change to 1 if your images have low intensity and you wish to enhance contrast 
    "top_thresh": 98, #this is gonna be the percentile threshold for images with higher background, adjust it based on your settings
    "mid_thresh": 98,#this is gonna be the percentile threshold for images with medium background, adjust it based on your settings
    "low_thresh": 97, #this is gonna be the percentile threshold for images with lower background, adjust it based on your settings
    "extra_bright_thresh": 98.5, #extra threshold, if you
    "high_int_thresh": 90, #are your images super bright? Then increase this value
    "low_int_thresh": 98.8 #are your images super holey? Then decrease this value
}

#%% path to images
#change this to the path to your images 
path="/Users/romina/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/Y1/internship_1/remote_memory_julia"
# %% this is gonna print all of the files in the path directory
all_files=[]
for filename in os.listdir(path):
    if filename.endswith(".tif"):
        all_files.append(filename)
    if filename.endswith(".nd2"):
        all_files.append(filename)
print(all_files)

#%% get intensity cut off values for filtering function
fos_ints = intensitySaver(path, all_files, channel, fos_thresh.get("is_intensity_low")).getIntensityValues()
low_int, high_int, int_cutoff_up, int_cutoff, int_cutoff_down= fos_ints.values()
# %%
counts=pd.DataFrame(columns=["img_ID","fos_cells"])
for filename in all_files:
    try:
        name= path + "/" + filename
        fos, stacks=getImg(channel, name) 
        blobs= getCoords(fos, stacks, circ, axis_ratio, axis_min, axis_limit).coordsCells(fos_thresh, fos_ints)
        overlap=getOverlap(stacks, dist_parameter).overlap_coords(blobs)
        fos_count=len(blobs)-len(overlap)
        count_list=(filename, fos_count)
        c_series = pd.Series(count_list, index = counts.columns)
        counts = counts.append(c_series, ignore_index=True)
    except:
        print(filename + " not_read")

# %%
counts.to_csv("/Users/romina/Desktop/counts_fos_julia_remmem.csv") #add the path and name of the results file
