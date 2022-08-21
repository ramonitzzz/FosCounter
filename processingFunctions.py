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

def show_original_filt (original, filt):
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(original, cmap="gray_r")
    ax2.imshow(filt, cmap="gray_r")
    ax1.set_axis_off()
    ax2.set_axis_off()
    plt.tight_layout()

def show_labels(img, img_original, circ, axis_min, axis_limit, axis_ratio):
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

class getThresh:
    def __init__(self, img, top_thresh, low_thresh):
        self.img= img
        self.top_thresh= top_thresh
        self.low_thresh= low_thresh


    def threshPV(self):
        denoise=sk.restoration.denoise_wavelet(self.img)
        blurred = sk.filters.gaussian(denoise, sigma=2.0)
        clahe= sk.exposure.equalize_adapthist(blurred, kernel_size=127,clip_limit=0.01,  nbins=256)
        t_thresh=sk.filters.threshold_otsu(clahe)
        if np.percentile(clahe, 25)<0.24:
            thresh=t_thresh+np.percentile(clahe, 80)
        else:
            if t_thresh/np.median(clahe)<=1.45:
                if t_thresh/np.median(clahe)<=1.02: 
                    if t_thresh/np.median(clahe)<=0.88:
                        thresh=t_thresh + np.percentile(clahe, 60)
                    else:
                        thresh=t_thresh + np.percentile(clahe, 18.5)
                else:
                    thresh= t_thresh + np.percentile(clahe, self.top_thresh)
            else:
                thresh= t_thresh + np.percentile(clahe, self.low_thresh)
        filtPV= filteredImg(clahe, thresh)
        return filtPV
    
    def threshFos(self):
        denoise=sk.restoration.denoise_wavelet(self.img)
        blurred = sk.filters.gaussian(denoise, sigma=2.0)
        t_thresh=sk.filters.threshold_otsu(blurred)
        if t_thresh/np.median(blurred)<=1.5:
            thresh= t_thresh + np.percentile(blurred, self.top_thresh)
        elif t_thresh/np.median(blurred)>=3:
            thresh= t_thresh + np.percentile(blurred, 20)
        else:
            thresh= t_thresh + np.percentile(blurred,self.low_thresh)
        filtFos= filteredImg(blurred, thresh)
        return filtFos
    
    def threshMC(self):
        denoise=sk.restoration.denoise_wavelet(self.img)
        blurred = sk.filters.gaussian(denoise, sigma=2.0)
        t_thresh=sk.filters.threshold_otsu(blurred)
        if np.percentile(blurred, 99)<= 0.0038: 
            thresh= t_thresh + np.percentile(blurred, 95) #change 95 for 90
        else:
            if t_thresh/np.median(blurred)>=4:
                thresh= t_thresh + np.percentile(blurred, 25)
            else:
                if t_thresh/np.median(blurred)<=1.5: 
                    if t_thresh/np.median(blurred)<=1:
                        thresh=t_thresh + np.percentile(blurred, 98)
                    else:
                        thresh= t_thresh + np.percentile(blurred, self.top_thresh)
                else:
                    thresh= t_thresh + np.percentile(blurred,self.low_thresh)
        filtMC= filteredImg(blurred, thresh)
        return filtMC


class getCoords:
    def __init__ (self, img, stacks, circ, axis_ratio, axis_min, axis_limit):
        self.img= img
        self.stacks= stacks
        self.circ= circ
        self.axis_ratio = axis_ratio
        self.axis_min= axis_min
        self.axis_limit= axis_limit

    def coords(self, filt, i):
        blobs_coords=pd.DataFrame(columns=["x","y","z"])
        labels = sk.measure.label(filt)
        props = sk.measure.regionprops_table(labels, properties=('centroid','axis_major_length','axis_minor_length', 'bbox', 'equivalent_diameter_area','label', 'eccentricity'))
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
  

    def coordsPV(self, top_thresh, low_thresh):
        blobs=pd.DataFrame(columns=["x","y","z"])
        for i in range(self.stacks):
            img_c=self.img[i]
            filt=getThresh(img_c, top_thresh, low_thresh).threshPV()
            blobs_coords=self.coords(filt, i)
            blobs=pd.concat([blobs,blobs_coords], ignore_index=True)
        return blobs
    
    def coordsFos(self, top_thresh, low_thresh):
        blobs=pd.DataFrame(columns=["x","y","z"])
        for i in range(self.stacks):
            img_c=self.img[i]
            filt=getThresh(img_c, top_thresh, low_thresh).threshFos()
            blobs_coords=self.coords(filt, i)
            blobs=pd.concat([blobs,blobs_coords], ignore_index=True)
        return blobs
    
    def coordsMC(self, top_thresh, low_thresh):
        blobs=pd.DataFrame(columns=["x","y","z"])
        for i in range(self.stacks):
            img_c=self.img[i]
            filt=getThresh(img_c, top_thresh, low_thresh).threshMC()
            blobs_coords=self.coords(filt, i)
            blobs=pd.concat([blobs,blobs_coords], ignore_index=True)
        return blobs

class getOverlap:
    def __init__(self, stacks, dist_thresh):
        self.stacks=stacks
        self.dist_threshr=dist_thresh
    
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
    def __init__(self, path, files, channel):
        self.path=path
        self.files=files
        self.channel= channel

    def getInts(self):
        ints= pd.DataFrame(columns=["25 percentile", "median", "99 percentile", "thresh/median"])
        for filename in self.files:
            name= self.path + "/" + filename
            fos, stacks=getImg(self.channel, name)
            for i in range(stacks):
                denoise=sk.restoration.denoise_wavelet(fos[i])
                blurred = sk.filters.gaussian(denoise, sigma=2.0)
                thresh=sk.filters.threshold_otsu(blurred)
                p25= np.percentile(blurred, 25)
                med=np.median(blurred)
                p99= np.percentile(blurred, 99)
                th_med= thresh/np.median(blurred)
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
        
        return low_int, high_int, int_cutoff_up, int_cutoff, int_cutoff_down