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

#%%
def getImg(channel, data_path):
    if ".nd2" in data_path:
        vol= nd2.imread(data_path)
    else:
        vol=imageio.volread(data_path)
    img=vol[:,channel,:,:]
    stacks= np.shape(img)[0]
    return img, stacks
 
def filteredImg(prepro, thresh):
        cells=np.where(prepro >=thresh, 1, 0)
        filtered=ndi.median_filter(cells, size=5)
        eroded=ndi.binary_erosion(filtered)
        dilated= ndi.binary_dilation(eroded, iterations=1)
        eroded=ndi.binary_erosion(dilated, iterations=2)
        filt=ndi.median_filter(eroded, size=5)
        return filt

def removeAxons(img, min_distance=10):
    distance = ndi.distance_transform_edt(img)

    local_max_coords = sk.feature.peak_local_max(distance, min_distance, num_peaks_per_label=2)
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True
    markers = sk.measure.label(local_max_mask)

    segNeu = sk.segmentation.watershed(-distance, markers, mask=img) #segmented neuron

    return segNeu

def show_original_filt (original, filt):
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(original, cmap="gray_r")
    ax2.imshow(filt, cmap="gray_r")
    ax1.set_axis_off()
    ax2.set_axis_off()
    plt.tight_layout()

def show_labels(img, img_original, circ, axis_min, axis_limit, axis_ratio, remove_axon=None):
    if remove_axon is None:
        label_image= sk.measure.label(img)
    else:
        label_image=removeAxons(img)
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

class getThresh:
    def __init__(self, img):
        self.img= img

    def thresh(self, thresh_dict, int_dict):
        denoise=sk.restoration.denoise_wavelet(self.img)
        blurred = sk.filters.gaussian(denoise, sigma=2.0)
        if thresh_dict.get("is_intensity_low") ==0:
            prepro=blurred
        else:
            prepro= sk.exposure.equalize_adapthist(blurred, kernel_size=127,clip_limit=0.01,  nbins=256)
        t_thresh=sk.filters.threshold_otsu(prepro)
        #filter images with too low or too bright intensities 
        if np.percentile(prepro, 25)<= int_dict.get("low_int"): #this is a safety net for images with too little intensity, you can try and adjust this if needed
            thresh= t_thresh + np.percentile(prepro, thresh_dict.get("high_int_thresh")) 
        elif np.percentile(prepro,99)>= int_dict.get("high_int"):
            thresh= t_thresh + np.percentile(prepro, thresh_dict.get("low_int_thresh")) #this is a safety net for images with too much intensity, you can also try and adjust this if needed 
        else:
            if t_thresh/np.median(prepro)>=int_dict.get("int_cutoff_up"):
                thresh= t_thresh + np.percentile(prepro, thresh_dict.get("extra_bright_thresh"))
            else:
                if t_thresh/np.median(prepro)<=int_dict.get("int_cutoff"): 
                    if t_thresh/np.median(prepro)<=int_dict.get("int_cutoff_down"): #before int_cutoff*0.66
                        thresh=t_thresh + np.percentile(prepro, thresh_dict.get("low_thresh"))
                    else:
                        thresh= t_thresh + np.percentile(prepro, thresh_dict.get("top_thresh"))
                else:
                    thresh= t_thresh + np.percentile(prepro, thresh_dict.get("mid_thresh"))
        filtFos=filteredImg(prepro, thresh)
        return filtFos

    def intInfo(self, stacks, int_dict, is_intensity_low=0):
        intInfo= pd.DataFrame(columns=["stack","classifier value", "25 p", "99 p", "threshold applied"])
        for i in range(stacks):
            denoise=sk.restoration.denoise_wavelet(self.img[i])
            blurred = sk.filters.gaussian(denoise, sigma=2.0)
            if is_intensity_low ==0:
                prepro=blurred
            else:
                prepro= sk.exposure.equalize_adapthist(blurred, kernel_size=127,clip_limit=0.01,  nbins=256)
            thresh=sk.filters.threshold_otsu(prepro)
            if np.percentile(prepro, 25)<= int_dict.get("low_int"): 
                cat= "high_int_thresh"
            elif np.percentile(prepro,99)>= int_dict.get("high_int"):
                cat= "low_int_thresh"
            else:
                if thresh/np.median(prepro)>=int_dict.get("int_cutoff_up"):
                    cat= "extra_bright_thresh"
                else:
                    if thresh/np.median(prepro)<=int_dict.get("int_cutoff"): 
                        if thresh/np.median(prepro)<=int_dict.get("int_cutoff_down"): 
                            cat= "low_thresh"
                        else:
                            cat= "top thresh"
                    else:
                        cat= "mid thresh"
            val_list= [i, thresh/np.median(prepro),np.percentile(prepro, 25), np.percentile(prepro, 99), cat]
            c_series = pd.Series(val_list, index = intInfo.columns)
            intInfo = intInfo.append(c_series, ignore_index=True)
        return intInfo
 


class getCoords:
    def __init__ (self, img, stacks, circ, axis_ratio, axis_min, axis_limit, remove_axon=None):
        self.img= img
        self.stacks= stacks
        self.circ= circ
        self.axis_ratio = axis_ratio
        self.axis_min= axis_min
        self.axis_limit= axis_limit
        self.remove_axon= remove_axon

    def coords(self, filt, i):
        blobs_coords=pd.DataFrame(columns=["x","y","z"])
        if self.remove_axon is None:
            labels= sk.measure.label(filt)
        else:
            labels= removeAxons(filt)
        #labels = sk.measure.label(filt)
        #props = sk.measure.regionprops_table(labels, properties=('centroid','axis_major_length','axis_minor_length', 'bbox', 'equivalent_diameter_area','label', 'eccentricity'))
        props = sk.measure.regionprops_table(labels, properties=('centroid','axis_major_length','axis_minor_length', 'bbox', 'equivalent_diameter_area','label', 'eccentricity',), cache=False)
        props_table=pd.DataFrame(props)
        props_filtered = props_table[props_table['eccentricity'] <= self.circ]
        #props_filtered = props_table[props_table['axis_minor_length'] >= axis_min]
        props_filtered=props_filtered[props_filtered['axis_minor_length'] >= self.axis_min]
        props_filtered=props_filtered[props_filtered['axis_major_length'] <= self.axis_limit]
        props_filtered=props_filtered[(props_filtered['axis_minor_length']/props_filtered['axis_major_length']) >= self.axis_ratio]
        #props_filtered=props_filtered[props_filtered['eccentricity'] <= circ]
        blobs_coords["x"]=props_filtered["centroid-0"]
        blobs_coords["y"]=props_filtered["centroid-1"]
        blobs_coords["z"]=i
        return blobs_coords

    
    def coordsCells(self, thresh_dict, int_dict):
        blobs=pd.DataFrame(columns=["x","y","z"])
        for i in range(self.stacks):
            img_c=self.img[i]
            filt=getThresh(img_c).thresh(thresh_dict, int_dict)
            blobs_coords=self.coords(filt, i)
            blobs=pd.concat([blobs,blobs_coords], ignore_index=True)
        return blobs

class getOverlap:
    def __init__(self, stacks, dist_thresh):
        self.stacks=stacks
        self.dist_thresh=dist_thresh
    
    #overlapping coordinates between z-stacks of one cell type
    def overlap_coords(self, blobs):
        overlap=pd.DataFrame(columns=["x1","y1","z1","x2","y2","z2","dist"])
        for i in range(self.stacks):
            for index, row in blobs.iterrows():
                if row["z"]==i:
                    for index_2, row_2 in blobs.iterrows():
                        if row_2["z"]== (i+1):
                            #dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                            dist= math.sqrt((row_2["x"]-row["x"])**2 + (row_2["y"]-row["y"])**2)
                            if dist< self.dist_thresh:
                                ov_list=(row["x"], row["y"], row["z"],row_2["x"],row_2["y"],row_2["z"],dist)
                                a_series = pd.Series(ov_list, index = overlap.columns)
                                overlap = overlap.append(a_series, ignore_index=True)
        return overlap

    #overlap between two cell types
    def overlap_cells(self, blobs1, blobs2):
        overlap=pd.DataFrame(columns=["x1","y1","z1","x2","y2","z2","dist"])
        for i in range(self.stacks):
            for index, row in blobs1.iterrows():
                if row["z"]==i:
                    for index_2, row_2 in blobs2.iterrows():
                        if row_2["z"]== (i):
                            #dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                            dist= math.sqrt((row_2["x"]-row["x"])**2 + (row_2["y"]-row["y"])**2)
                            if dist< self.dist_thresh:
                                ov_list=(row["x"], row["y"], row["z"],row_2["x"],row_2["y"],row_2["z"],dist)
                                a_series = pd.Series(ov_list, index = overlap.columns)
                                overlap = overlap.append(a_series, ignore_index=True)
        return overlap

    #overlap of overlap
    def overlap_cells_img(self, ov_cells):
        overlap=pd.DataFrame(columns=["x1","y1","z1","x2","y2","z2","dist"])
        for i in range(self.stacks):
            for index, row in ov_cells.iterrows():
                if row["z1"]==i:
                    for index_2, row_2 in ov_cells.iterrows():
                        if row_2["z1"]== (i+1):
                            #dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                            dist= math.sqrt((row_2["x1"]-row["x1"])**2 + (row_2["y1"]-row["y1"])**2)
                            if dist< self.dist_thresh:
                                ov_list=(row["x1"], row["y1"], row["z1"],row_2["x1"],row_2["y1"],row_2["z1"],dist)
                                a_series = pd.Series(ov_list, index = overlap.columns)
                                overlap = overlap.append(a_series, ignore_index=True)
        return overlap

    #overlap of three types of cells
    def overlap_all(self, ov12, blobs3):
        overlap=pd.DataFrame(columns=["x1","y1","z1","x2","y2","z2","dist","x3","y3","z3", "dist23"]) #dist is dist12
        for i in range(self.stacks):
            for index, row in ov12.iterrows():
                if row["z1"]==i:
                    for index_2, row_2 in blobs3.iterrows():
                        if row_2["z"]== (i):
                            #dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                            dist= math.sqrt((row_2["x"]-row["x1"])**2 + (row_2["y"]-row["y1"])**2)
                            if dist< self.dist_thresh:
                                ov_list=(row["x1"], row["y1"], row["z1"],row["x2"],row["y2"],row["z2"],row["dist"],row_2["x"],row_2["y"],row_2["z"],dist)
                                a_series = pd.Series(ov_list, index = overlap.columns)
                                overlap = overlap.append(a_series, ignore_index=True)
        return overlap
    
class intensitySaver:
    def __init__(self, path, files, channel, is_intensity_low):
        self.path=path
        self.files=files
        self.channel= channel
        self.is_intensity_low= is_intensity_low

    def getInts(self):
        ints= pd.DataFrame(columns=["25 percentile", "median", "99 percentile", "thresh/median"])
        for filename in self.files:
            name= self.path + "/" + filename
            fos, stacks=getImg(self.channel, name)
            for i in range(stacks):
                denoise=sk.restoration.denoise_wavelet(fos[i])
                blurred = sk.filters.gaussian(denoise, sigma=2.0)
                if self.is_intensity_low ==0:
                    prepro=blurred
                else:
                    prepro= sk.exposure.equalize_adapthist(blurred, kernel_size=127,clip_limit=0.01,  nbins=256)
                thresh=sk.filters.threshold_otsu(prepro)
                p25= np.percentile(prepro, 25)
                med=np.median(prepro)
                p99= np.percentile(prepro, 99)
                th_med= thresh/np.median(prepro)
                val_list=(p25, med, p99, th_med)
                c_series = pd.Series(val_list, index = ints.columns)
                ints= ints.append(c_series, ignore_index=True)
        return ints

    def getIntensityValues(self):
        ints=self.getInts()
        #get int cut off values 
        low_int= ints["25 percentile"].quantile(0.05)
        high_int=ints["99 percentile"].quantile(0.95)
        int_cutoff= ints["thresh/median"].median()/2
        int_cutoff_up= ints["thresh/median"].quantile(0.85)
        int_cutoff_down= ints["thresh/median"].quantile(0.15)

        #reorder intensity cut off points if necessary
        if int_cutoff_down> int_cutoff:
            saver=int_cutoff_down
            int_cutoff_down=int_cutoff
            int_cutoff=saver

        if int_cutoff_up< int_cutoff:
            saver= int_cutoff
            int_cutoff=int_cutoff_up
            int_cutoff_up= saver
        
        #make dictionary

        int_dic={
            "low_int": low_int,
            "high_int": high_int,
            "int_cutoff_up": int_cutoff_up,
            "int_cutoff": int_cutoff,
            "int_cutoff_down": int_cutoff_down
        }
        return int_dic