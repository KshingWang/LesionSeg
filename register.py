import ants
import os
path = '~/Documents/normalized_ISBI2015/challenge'
path1 = '~/Documents/normalized_ISBI2015/before_registration'
fi = ants.image_read(os.path.join(path,'norm_01_01_t1.nii'))
mi = ants.image_read(os.path.join(path1,'norm_01_01_flair.nii'))
trans = ants.resample_image_to_target(mi,fi,verbose = True)
regi = ants.registration(fixed = fi, moving = mi, type_of_transform = 'Rigid')
ants.image_write(regi['warpedmovout'],os.path.join(path,'norm_01_01_flair.nii'))