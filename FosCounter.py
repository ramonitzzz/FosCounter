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
from processingFunctions import show_original_filt, show_labels, filteredImg
from processingFunctions import getImg, intensitySaver

#%% <functions>
def filtering(img, top_thresh, mid_thresh, low_thresh, high_int_thresh, low_int_thresh):
    denoise=sk.restoration.denoise_wavelet(img)
    blurred = sk.filters.gaussian(denoise, sigma=2.0)
    if is_intensity_low ==0:
        prepro=blurred
    else:
        prepro= sk.exposure.equalize_adapthist(blurred, kernel_size=127,clip_limit=0.01,  nbins=256)
    t_thresh=sk.filters.threshold_otsu(prepro)
    #filter images with too low or too bright intensities 
    if np.percentile(prepro, 25)<= low_int: #this is a safety net for images with too little intensity, you can try and adjust this if needed
        thresh= t_thresh + np.percentile(prepro, high_int_thresh) 
    elif np.percentile(prepro,99)>= high_int:
        thresh= t_thresh + np.percentile(prepro, low_int_thresh) #this is a safety net for images with too much intensity, you can also try and adjust this if needed 
    else:
        if t_thresh/np.median(prepro)>=int_cutoff_up:
            thresh= t_thresh
        else:
            if t_thresh/np.median(prepro)<=int_cutoff: 
                if t_thresh/np.median(prepro)<=int_cutoff_down: #before int_cutoff*0.66
                    thresh=t_thresh + np.percentile(prepro, low_thresh)
                else:
                    thresh= t_thresh + np.percentile(prepro, top_thresh)
            else:
                thresh= t_thresh + np.percentile(prepro, mid_thresh)
    fos_cells=np.where(prepro >= thresh, 1, 0)
    filtered=ndi.median_filter(fos_cells, size=5)
    eroded=ndi.binary_erosion(filtered)
    dilated= ndi.binary_dilation(eroded, iterations=1)
    eroded=ndi.binary_erosion(dilated, iterations=2)
    filt=ndi.median_filter(eroded, size=5)
    return filt

def blobs_coordinates(img, stacks):
    blobs=pd.DataFrame(columns=["x","y","z"])
    for i in range(stacks):
        blobs_coords=pd.DataFrame(columns=["x","y","z"])
        fos_c=img[i]
        filt=filtering(fos_c, top_thresh, mid_thresh, low_thresh, high_int_thresh, low_int_thresh)
        labels = sk.measure.label(filt)
        props = sk.measure.regionprops_table(labels, properties=('centroid','axis_major_length','axis_minor_length', 'bbox', 'equivalent_diameter_area','label', 'eccentricity'))
        props_table=pd.DataFrame(props)
        props_filtered = props_table[props_table['eccentricity'] <= circ]
        #props_filtered = props_table[props_table['axis_minor_length'] >= axis_min]
        props_filtered=props_filtered[props_filtered['axis_minor_length'] >= axis_min]
        props_filtered=props_filtered[props_filtered['axis_major_length'] <= axis_limit]
        props_filtered=props_filtered[(props_filtered['axis_minor_length']/props_filtered['axis_major_length']) >= axis_ratio]
        #props_filtered=props_filtered[props_filtered['eccentricity'] <= circ]
        blobs_coords["x"]=props_filtered["centroid-0"]
        blobs_coords["y"]=props_filtered["centroid-1"]
        blobs_coords["z"]=i
        blobs=pd.concat([blobs,blobs_coords], ignore_index=True)
    return blobs

def overlap_coords(blobs, stacks, dist_thresh):
    overlap=pd.DataFrame(columns=["x1","y1","z1","x2","y2","z2","dist"])
    for i in range(stacks):
        for index, row in blobs.iterrows():
            if row["z"]==i:
                for index_2, row_2 in blobs.iterrows():
                    if row_2["z"]== (i+1):
                        #dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        dist= math.sqrt((row_2["x"]-row["x"])**2 + (row_2["y"]-row["y"])**2)
                        if dist<dist_thresh:
                            ov_list=(row["x"], row["y"], row["z"],row_2["x"],row_2["y"],row_2["z"],dist)
                            a_series = pd.Series(ov_list, index = overlap.columns)
                            overlap = overlap.append(a_series, ignore_index=True)
    return overlap


#%% PARAMETERS
# you're gonna have to adjust the thresholding parameters depending on your images 
is_intensity_low=0 #change to 1 if your images have low intensity and you wish to enhance contrast
axis_limit = 60
dist_parameter=25
axis_min=10.5
circ=0.98
axis_ratio= 0.45
top_thresh=95 #this is gonna be the threshold for images with higher background, adjust it based on your settings
mid_thresh=92 #this is gonna be the threshold for images with medium background, adjust it based on your settings
low_thresh=90 #this is gonna be the threshold for images with lower background, adjust it based on your settings
high_int_thresh=98
low_int_thresh=4

#%% path to images
path="/PathGoesHere/fos_images" #change this to the path to your images 

# %% this is gonna print all of the files in the path directory
all_files=[]
for filename in os.listdir(path):
    if filename.endswith(".tif"):
        all_files.append(filename)
    if filename.endswith(".nd2"):
        all_files.append(filename)
print(all_files)

#%% get intensity cut off values for filtering function
low_int, high_int, int_cutoff_up, int_cutoff, int_cutoff_down = intensitySaver(path, all_files, 1).getIntensityValues()
# %%
counts=pd.DataFrame(columns=["img_ID","fos_cells"])
for filename in all_files:
    try:
        name= path + "/" + filename
        fos, stacks=getImg(3, name) #change the 3 here for the channel where your fos is stained, note that the count starts in 0 so if you have fos in the first channel of your image you need to put 0 here and so on (i.e here the 3 corresponds to the IR channel)
        blobs= blobs_coordinates(fos, stacks)
        overlap=overlap_coords(blobs, stacks, dist_parameter)
        fos_count=len(blobs)-len(overlap)
        count_list=(filename, fos_count)
        c_series = pd.Series(count_list, index = counts.columns)
        counts = counts.append(c_series, ignore_index=True)
    except:
        print(filename + " not_read")

# %%
counts.to_csv("/PathGoesHere/counts_fos.csv") #add the path and name of the results file

########
#here is where you can see your images and try which threshold suits you best 
 # try adjusting the top and low threshold values until you get optimal results
# %% 
data= path + "/"+"35_1.tif" #open an image here to examinate (note that you're gonna have to open a few to see how your images vary from each other)
fos_t, stacks=getImg(3, data)
print(stacks)

# %% here you can see the values of the image intensities 
for i in range(stacks):
    denoise=sk.restoration.denoise_wavelet(fos_t[i])
    blurred = sk.filters.gaussian(denoise, sigma=2.0)
    thresh=sk.filters.threshold_otsu(blurred)
    print(thresh/np.median(blurred)) #note that this is what the conditional thresholding is based on, this is the relation to the otsu threshold method to the median value of the intensity of the image
    print("25 p:", np.percentile(blurred, 25), "99 p:", np.percentile(blurred, 99)) #this will give you information about the range of the image intensity

# %% TEST TRIAL 
#here you can visually inspect the image and how it is being thresholded, and then adjust your thresholding values accordingly 
for i in range(stacks):
    fos_cc=fos_t[i]
    filt=filtering(fos_cc, top_thresh, mid_thresh, low_thresh, high_int_thresh, low_int_thresh)
    show_labels(filt, fos_cc, circ, axis_min, axis_limit, axis_ratio)

# %% here you can see the fos count of the trial image 
blobs= blobs_coordinates(fos_t, stacks)
overlap=overlap_coords(blobs, stacks, dist_parameter)
print(len(blobs)-len(overlap))
