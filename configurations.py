import os

# TODO: change the following path based on your dataset
<<<<<<< Updated upstream
PATH_DATASET = os.path.join(os.path.expanduser('~'), 'Documents', 'Datasets', 'sample_dataset')
=======
PATH_DATASET = os.path.join('/data/h_oguz_lab/wanj119', '7T_lesion_seg')
>>>>>>> Stashed changes

# We assume the format of file names in this pattern {PREFIX}_{PATIENT-ID}_{TIMEPOINT-ID}_{MODALITY(MASK)}.{SUFFIX}
# PREFIX: can be any string
# PATIENT-ID: number id of the patient
# TIMEPOINT-ID: number id of the timepoint
# MODALITY, MASK: t1, flair, t2, pd, mask1, mask2, etc
# SUFFIX: nii, nii.gz, etc
# e.g. training_01_01_flair.nii, training_03_05_mask1.nii
# TODO: change the following constants based on your dataset
<<<<<<< Updated upstream
MODALITIES = ['t1', 'flair', 't2', 'pd']
MASKS = ['mask1', 'mask2']
SUFFIX = 'nii'
=======
#MODALITIES = ['t1', 'flair', 't2', 'pd']
#MODALITIES = ['swi', 'flair', 'fi']
MODALITIES = ['t1', 'flair']
#MODALITIES = ['flair', 'flairswi', 'phase','qsm','r2']

#MASKS = ['mask','mask1', 'mask2']
MASKS = ['label']
SUFFIX = 'nii.gz'
>>>>>>> Stashed changes

# The axis corresponding to axial, sagittal and coronal, respectively
# TODO: change the following axes based on your dataset
AXIS_TO_TAKE = [2, 0, 1]