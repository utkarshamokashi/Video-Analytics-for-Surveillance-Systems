import h5py
import numpy as np
import os
import cv2
import glob
from pathlib import Path
import tensorflow as tf

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
'''This function takes the files with the same name from LR and HR folders and stores the new dataset in h5 format'''
#create LR and HR image lists
LRimages = Path("C:\\Users\\Aarush\\Desktop\\Files\\Project\\Video\\LR").glob("*.jpg")
HRimages = Path("C:\\Users\\Aarush\\Desktop\\Files\\Project\\Video\\HR").glob("*.jpg")
#sort the lists
LR_images = [str(p) for p in LRimages]
HR_images = [str(p) for p in HRimages]
LR_images.sort()
HR_images.sort()
print('LR_images: ',LR_images)
print('HR_images: ',HR_images)
#create a dataset in the h5 file
# get LR image dtype and create a h5 file
LR_dt = cv2.imread(LR_images[0]).dtype
with h5py.File('our_dataset.h5','w') as h5_file:
    #create a dataset in the h5 file
    dataset = h5_file.create_dataset('our_dataset',(len(LR_images),2,360,640,3),dtype=LR_dt)
    #store the images in the dataset
    for i in range(len(LR_images)):
        LR_image = cv2.imread(LR_images[i])
        HR_image = cv2.imread(HR_images[i])
        dataset[i,0,0:360,0:640,:] = LR_image
        dataset[i,1,:,:,:] = HR_image
h5_file.close()