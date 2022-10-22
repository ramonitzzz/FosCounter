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
#%% dataset parameters
channelFos=1
channelPV=0
channelMC=2
axis_limit = 60
dist_thresh=25
circ=0.98

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
counts=pd.DataFrame(columns=["img_ID","fos_cells", "pv_cells", "mCherry", "pv+fos", "pv+mCherry", "fos+mCherry", "fos+pv+mCherry"])
for filename in all_files:
    name= path + "/"+ filename
    fos, stacks=getImg(channelFos, name)
    pv, s2=getImg(channelPV, name)
    mc, s3= getImg(channelMC, name)
    #fos cells
    blobs_fos= getCoords(fos, stacks, circ, 0.45, 11, axis_limit).coordsFos(fos_thresh, fos_ints)
    overlap_fos=getOverlap(stacks, dist_thresh).overlap_coords(blobs_fos)
    fos_count=len(blobs_fos)-len(overlap_fos)
    #pv cells
    blobs_pv= getCoords(pv, stacks, circ, 0.4, 14, axis_limit).coordsPV(50, 20)
    overlap_pv=getOverlap(stacks, dist_thresh).overlap_coords(blobs_pv)
    pv_count=len(blobs_pv)-len(overlap_pv)
    #mCherry cells
    blobs_mc= getCoords(mc, stacks, circ, 0.5, 14, axis_limit).coordsMC(90, 85)
    overlap_mc=getOverlap(stacks, dist_thresh).overlap_coords(blobs_mc)
    mc_count=len(blobs_mc)-len(overlap_mc)
    
    #pv + fos
    fos_pv= getOverlap(stacks, dist_thresh).overlap_cells(blobs_fos, blobs_pv)
    overlap_coords_fos_pv=getOverlap(stacks, 10).overlap_cells_img(fos_pv)
    counts_fos_pv= len(fos_pv)-len(overlap_coords_fos_pv)
    #pv + mc
    pv_mc= getOverlap(stacks, dist_thresh).overlap_cells(blobs_pv, blobs_mc)
    overlap_coords_pv_mc=getOverlap(stacks, 10).overlap_cells_img(pv_mc)
    counts_pv_mc= len(pv_mc)-len(overlap_coords_pv_mc)
    # fos + mc
    fos_mc= getOverlap(stacks, dist_thresh).overlap_cells(blobs_fos, blobs_mc)
    overlap_coords_fos_mc=getOverlap(stacks, 10).overlap_cells_img(fos_mc)
    counts_fos_mc= len(fos_mc)-len(overlap_coords_fos_mc)
    # fos + pv + mc
    fos_pv_mc= getOverlap(stacks, 10).overlap_all(fos_pv, blobs_mc)
    overlap_coords_fos_pv_mc= getOverlap(stacks,10).overlap_cells_img(fos_pv_mc)
    counts_fos_pv_mc= len(fos_pv_mc)-len(overlap_coords_fos_pv_mc)
    #make dataframe
    count_list=(filename, fos_count, pv_count, mc_count, counts_fos_pv, counts_pv_mc, counts_fos_mc, counts_fos_pv_mc)
    c_series = pd.Series(count_list, index = counts.columns)
    counts = counts.append(c_series, ignore_index=True)

#%%
counts.to_csv("/Users/romina/Desktop/counts_fos_pv_mcherry_retrieval_mc_v2.csv")

