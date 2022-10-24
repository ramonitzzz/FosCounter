#%% <import modules>
import skimage as sk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import pandas as pd
import math
from scipy import ndimage as ndi
import imageio
import nd2
from processingFunctions import * 

#%% path to dataset
path="/Users/romina/Library/Y1/internship_1/remote_memory/NisslIR_mCherry_PVB_fosG"

#%% retrieve all files from dataset folder
all_files=[]
for filename in os.listdir(path):
    if filename.endswith(".tif"):
        all_files.append(filename)
    if filename.endswith(".nd2"):
        all_files.append(filename)
print(all_files)
#%% PARAMETERS
# you're gonna have to adjust the thresholding parameters depending on your images 
channelFos=1 #channel where Fos staining is
is_intensity_low=1 #change to 1 if your images have low intensity and you wish to enhance contrast
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

#%% run script and get counts
counts=pd.DataFrame(columns=["img_ID","fos_cells"])
for filename in all_files:
    try:
        name= path + "/" + filename
        fos, stacks=getImg(channel, name) 
        blobs= getCoords(fos, stacks, circ, axis_ratio, axis_min, axis_limit).coordsFos(fos_thresh, fos_ints)
        overlap=getOverlap(stacks, dist_thresh).overlap_coords(blobs)
        fos_count=len(blobs)-len(overlap)
        count_list=(filename, fos_count)
        c_series = pd.Series(count_list, index = counts.columns)
        counts = counts.append(c_series, ignore_index=True)
    except:
        print(filename + " not_read")

# %%
#counts.to_csv("/PathGoesHere/counts_fos.csv") #add the path and name of the results file
counts.to_csv("/Users/romina/Desktop/counts_fos_remmem.csv") #add the path and name of the results file