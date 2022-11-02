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
import warnings
warnings.filterwarnings('ignore')

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
#%% General parameters
channelFos=1 #channel where Fos staining is
channelMC=2 #channel where MC staining is
channelPV=0 #channel where PV staining is
axis_limit = 60
dist_thresh=25
axis_min=11
circ=0.95
axis_ratio= 0.48

axis_ratio_mc=0.3
axis_min_mc=14

axis_ratio_pv=0.4
axis_min_pv=14


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

#MC parameters
mc_thresh={
    "is_intensity_low":0, # don't use for mcherry image analysis 
    "extra_bright_thresh": 99,
    "top_thresh":55, #this is gonna be the threshold for images with higher background, adjust it based on your settings
    "mid_thresh":50,#this is gonna be the threshold for images with medium background, adjust it based on your settings
    "low_thresh":20, #this is gonna be the threshold for images with lower background, adjust it based on your settings
    "high_int_thresh":80,
    "low_int_thresh":30
}

#PV parameters
pv_thresh={
    "is_intensity_low":0, #leave at 0 for PV
    "extra_bright_thresh": 50,
    "top_thresh":55, #this is gonna be the threshold for images with higher background, adjust it based on your settings
    "mid_thresh":50,#this is gonna be the threshold for images with medium background, adjust it based on your settings
    "low_thresh":20, #this is gonna be the threshold for images with lower background, adjust it based on your settings
    "high_int_thresh":80,
    "low_int_thresh":30
}
#%% get intensity cut off values for fos filtering function
fos_ints= intensitySaver(path, all_files, channelFos, fos_thresh.get("is_intensity_low")).getIntensityValues()
print ("Fos intensity cutoffs: ",  fos_ints)

# get intensity cut off value for filtering function, for mc
mc_ints= intensitySaver(path, all_files, channelMC, 0).getIntensityValues()
print("MCherry intensity cutoffs: ", mc_ints)

# get intensity cut off value for filtering function, for pv
pv_ints=intensitySaver(path, all_files, channelMC, 0).getIntensityValues()
print ("PV intensity cutoffs: ",pv_ints)
os.system('say "your code has finished"');
#%% run script and get counts
counts=pd.DataFrame(columns=["img_ID","fos_cells", "pv_cells", "mCherry", "pv+fos", "pv+mCherry", "fos+mCherry", "fos+pv+mCherry"])
for filename in all_files:
    name= path + "/"+ filename
    fos, stacks=getImg(channelFos, name)
    pv, s2=getImg(channelPV, name)
    mc, s3= getImg(channelMC, name)
    #fos cells
    blobs_fos= getCoords(fos, stacks, circ, axis_ratio, axis_min, axis_limit).coordsCells(fos_thresh, fos_ints)
    overlap_fos=getOverlap(stacks, dist_thresh).overlap_coords(blobs_fos)
    fos_count=len(blobs_fos)-len(overlap_fos)
    #pv cells
    blobs_pv= getCoords(pv, stacks, circ, axis_ratio_pv, axis_min_pv, axis_limit).coordsCells(pv_thresh, pv_ints)
    overlap_pv=getOverlap(stacks, dist_thresh).overlap_coords(blobs_pv)
    pv_count=len(blobs_pv)-len(overlap_pv)
    #mCherry cells
    blobs_mc= getCoords(mc, stacks, circ, axis_ratio_mc, axis_min_mc, axis_limit, "remove axons").coordsCells(mc_thresh, mc_ints)
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

