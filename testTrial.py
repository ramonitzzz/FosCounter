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
warnings.filterwarnings('ignore')

#%% path to folder with the images
#path="/Users/juliavanadrichem/Desktop/Huidige telling/vtrap 3-5m_app_ps1"
path="/Users/juliavanadrichem/Desktop/Huidige telling/vtrap 3-5m_app_ps1"
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
## visualize how your images look after being thresholded with the previous parameter

#%% select one image to visualize if the thresholding is correct 
test_image= "vtrap_remote memory 3-5m_B.tif" #name of image file (incl tif) 
data= path + "/"+ test_image #open an image here to examinate (note that you're gonna have to open a few to see how your images vary from each other)
fos_t, stacks=getImg(channelFos, data) #if you want to analyse the mcherry channel, put channelMC
mc_t, stacks=getImg(channelMC, data)
pv_t, stacks=getImg(channelPV, data)
print(stacks) #will print the number of planes of the z-stack

#%% Fos 
#%% visually inspect the chosen image and how it is being thresholded for fos , and then adjust your thresholding values accordingly 
for i in range(stacks):
    fos_cc=fos_t[i]
    filt= getThresh(fos_cc).thresh(fos_thresh, fos_ints)
    show_labels(filt, fos_cc, circ, axis_min, axis_limit, axis_ratio)

#%%
# print the counts for Fos cells 
blobs_fos= getCoords(fos_t, stacks, circ, axis_ratio, axis_min, axis_limit).coordsCells(fos_thresh, fos_ints)
overlap=getOverlap(stacks, dist_thresh).overlap_coords(blobs_fos)
print(len(blobs_fos)-len(overlap))

#%%
#get values of each plane to assess where to adjust values 
intInfoFos= getThresh(fos_t).intInfo(stacks, fos_ints)
print(intInfoFos) #for Fos

#%% MC 
#select one image to visualize if the thresholding is correct 
#visually inspect the chosen image and how it is being thresholded for mc, and then adjust your thresholding values accordingly 
for i in range(stacks):
    mc_cc=mc_t[i]
    filt= getThresh(mc_cc).thresh(mc_thresh, mc_ints)
    show_labels(filt, mc_cc, circ, axis_min_mc, axis_limit, axis_ratio_mc, "remove axons") #delete "remove axons" argument if you have circular cells

#%%
# print the counts for MC cells 
blobs_mc=getCoords(mc_t, stacks, circ, axis_ratio_mc, axis_min_mc, axis_limit, "remove axons").coordsCells(mc_thresh, mc_ints) #delete "remove axons" argument if you have circular cells
overlap=getOverlap(stacks, dist_thresh).overlap_coords(blobs_mc)
print(len(blobs_mc)-len(overlap))

#%%get values of each plane to assess where to adjust values
#MCherry
intInfoMC= getThresh(mc_t).intInfo(stacks, mc_ints)
print(intInfoMC)

#%% PV 
# visually inspect the chosen image and how it is being thresholded for mc, and then adjust your thresholding values accordingly 
for i in range(stacks):
    pv_cc=pv_t[i]
    filt= getThresh(pv_cc).thresh(pv_thresh, pv_ints)
    show_labels(filt, pv_cc, circ, axis_min_mc, axis_limit, axis_ratio_mc)

#%%
# print the counts for PV cells 
blobs_pv=getCoords(pv_t, stacks, circ, axis_ratio_mc, axis_min_mc, axis_limit).coordsCells(pv_thresh, pv_ints)
overlap=getOverlap(stacks, dist_thresh).overlap_coords(blobs_pv)
print(len(blobs_pv)-len(overlap))

#%%get values of each plane to assess where to adjust values
#PV
intInfoPV= getThresh(pv_t).intInfo(stacks, pv_ints)
print(intInfoPV)

## once you have decided on the thresholding values copy them on the main file and run the script to obtain the counts for the dataset