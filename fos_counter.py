#%% <import modules>
import nd2
import skimage as sk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import pandas as pd
import math
from scipy import ndimage as ndi
import imageio

#%% <functions>
def fos_img(data_path):
    if ".nd2" in data_path:
        vol= nd2.imread(data_path)
    else:
        vol=imageio.volread(data_path)
    fos=vol[:,1,:,:]
    stacks= np.shape(fos)[0]
    return fos, stacks


def filtering(img):
    denoise=sk.restoration.denoise_wavelet(img)
    blurred = sk.filters.gaussian(denoise, sigma=2.0)
    clahe= sk.exposure.equalize_adapthist(blurred, kernel_size=127,clip_limit=0.01,  nbins=256)
    t_thresh=sk.filters.threshold_otsu(clahe)
    if t_thresh/np.median(clahe)<=1.45:
        thresh= t_thresh + np.percentile(clahe, top_thresh)
    else:
        thresh= t_thresh + np.percentile(clahe, low_thresh)
    fos_cells=np.where(clahe >=thresh, 1, 0)
    filtered=ndi.median_filter(fos_cells, size=5)
    eroded=ndi.binary_erosion(filtered)
    dilated= ndi.binary_dilation(eroded, iterations=1)
    eroded=ndi.binary_erosion(dilated, iterations=2)
    filt=ndi.median_filter(eroded, size=5)
    return filt

def show_original_filt (original, filt):
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(original, cmap="gray_r")
    ax2.imshow(filt, cmap="gray_r")
    ax1.set_axis_off()
    ax2.set_axis_off()
    plt.tight_layout()

def particle_analyzer (img):
    contours= sk.measure.find_contours(img, .8)
    label_image= sk.measure.label(img)
    regions = sk.measure.regionprops(label_image)
    #props = sk.measure.regionprops_table(label_image, properties=('eccentricity'))
    props = sk.measure.regionprops_table(label_image)
    props_table=pd.DataFrame(props)
    counts=len(contours)
    return props_table, counts


def show_labels(img, img_original):
    label_image= sk.measure.label(img)
    image_label_overlay = sk.color.label2rgb(label_image, image=img, bg_label=0)
    f, (ax1, ax2)=plt.subplots(1,2)
    #fig, ax2 = plt.subplots(figsize=(10, 6))
    ax1.imshow(img_original, cmap="gray_r")
    ax2.imshow(image_label_overlay, cmap="gray_r")

    for region in sk.measure.regionprops(label_image):
        # take regions with large enough areas
        if region.eccentricity<=circ:
            if region.axis_major_length <= axis_limit:
                if region.axis_minor_length >=axis_min:
                    if region.axis_minor_length / region.axis_major_length >=axis_ratio:
            # draw rectangle around segmented coins
                        minr, minc, maxr, maxc = region.bbox
                        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='red', linewidth=0.5)
                        ax2.add_patch(rect)


    ax1.set_axis_off()
    ax2.set_axis_off()
    plt.tight_layout()
    plt.show()

def blobs_coordinates(img, stacks):
    blobs=pd.DataFrame(columns=["x","y","z"])
    for i in range(stacks):
        blobs_coords=pd.DataFrame(columns=["x","y","z"])
        fos_c=img[i]
        filt=filtering(fos_c)
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

#%% <parameters>
axis_limit = 60
dist_parameter=20
axis_min=12
circ=0.98
axis_ratio= 0.5
top_thresh=50 #top thresh 50 for test images; 95 for recent mem ds
low_thresh=20 #low thresh 20 for test images; 91 for recent mem ds

#%% <remote memory images>
path="/Users/romina/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/internship_1/remote_memory/PV_IR_Fos_G_Nissl_B"

# %%
all_files=[]
for filename in os.listdir(path):
    if filename.endswith(".tif"):
        all_files.append(filename)
    if filename.endswith(".nd2"):
        all_files.append(filename)
print(all_files)

# %%
counts=pd.DataFrame(columns=["img_ID","fos_cells"])
for filename in all_files:
    try:
        name= path + "/" + filename
        fos, stacks=fos_img(name)
        blobs= blobs_coordinates(fos, stacks)
        overlap=overlap_coords(blobs, stacks, dist_parameter)
        fos_count=len(blobs)-len(overlap)
        count_list=(filename, fos_count)
        c_series = pd.Series(count_list, index = counts.columns)
        counts = counts.append(c_series, ignore_index=True)
    except:
        print(filename + " not read")
# %%
print(counts.head())
# %%
counts.to_csv("/Users/romina/Desktop/counts_fos_trial_2.csv")
# %%

