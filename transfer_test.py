# This function can help to normalize the intensity to 0-1
# By taking the (Intensiy - min)/(max - min)
# Edited in Sep 2021

# Test on transfer ISBI2015 data to 0-1

import nibabel as nib
import numpy as np
import os

def normalize_zerone(data_path):
    image = nib.load(data_path)
    data = image.get_fdata()
    normalize = np.array([(x - np.amin(data)) / (np.amax(data) - np.amin(data)) for x in data])
    #normalize1 = np.array([(x - np.min(data)) / (np.max(data) - np.min(data)) for x in data])
    return nib.Nifti1Image(normalize, image.affine, image.header)

# with os.scandir('/home/jackywang/Documents/normalized_ISBI2015') as entries:
#     for entry in entries:
#         filepath = os.path.join('/home/jackywang/Documents/normalized_ISBI2015',entry)
#         normed = normalize_zerone(filepath)
#         nib.save(normed, os.path.join('/home/jackywang/Documents/processed_folder',entry))

path = '/home/jc/Documents/Datasets/MS_Data/Pre-processed/Simulated/FLAIR'
#print(os.listdir(path))
#normed,normed1 = normalize_zerone('/home/jackywang/Documents/ISBI_2015/training/training01/preprocessed/training01_01_flair_pp.nii')
#nib.save(normed,'test1.nii')
#nib.save(normed1,'test2.nii')
for file in os.listdir(path):
    normed = normalize_zerone(os.path.join(path, file))
    save_path = '/home/jc/Documents/Datasets/MS_Data/Pre-processed/Simulated/FLAIR_zero1'
    nib.save(normed, os.path.join(save_path, file))