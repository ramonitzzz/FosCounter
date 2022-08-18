# FosCounter

FosCounter is a Python-based semi-automated method for detecting and counting cFos positive cells in IHC staining confocal z-stack images. Traditional cFos counting is a very time-consuming process that is prone to interpersonal and intrapersonal bias. Given that the criterion of inclusion is based on Fos intensity values, FosCounter proposes an alternative way to count Fos+ cells in a more standardized and efficient manner.

## Accounting for background intensity variation
To address the normal variation in background intensity of confocal microscope images, the algorithm applies conditional thresholding to each separate plane in a z-stack image to account for background intensity variation and set a minimum intensity threshold to exclude cells exhibiting a low Fos expression.

## Accounting for tridimensionality
The algorithm calculates the XY coordinates of the centroid of each cell with an intensity above the threshold value and within the size and shape criterion. Then, it computes the difference of coordinates between the subsequent step of the z-stack in order to exclude cells that are incorrectly counted more than once. 

## Results validation
The accuracy of the cFos cell counts obtained by the algorithm were validated by conducting a pairwise comparison of the Fos+ cells counted by the code against two datasets of pre-counted Fos-stained images, one in the far-red channel and one in the green channel (p> 0.05). 


#cellCounter

cellCounter is an expansion to FosCounter aimed to count other types of cells (here we count PV interneurons and mCherry tagged cells) in addition to cFos positive cells. The code also counts co-localization of the IHC stained cell types. 
