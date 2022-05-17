from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import nibabel as nib
import pickle


# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    ind = image_numpy.shape[0] // 2
    image_numpy = image_numpy[ind:ind + 1, :, :, ]
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    max_value, min_value = 2, -2
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) - min_value) / (max_value - min_value) * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def pkltonii(path, file):
    dir = os.path.join(path, file)
    with open(dir, 'rb') as f:
        data = pickle.load(f)
    # img = pickle.load(path)
    img = np.concatenate((data['t1'],data['t2'],data['flair']),axis=2)
    img = np.moveaxis(img,-1,0)
    print(img.shape)
    imgg = nib.Nifti1Image(img, np.eye(4))
    msk = data['mask']
    mskk = np.moveaxis(msk,-1,0)
    mask = nib.Nifti1Image(mskk, np.eye(4))

    nib.save(imgg, os.path.join('/home/hao/Documents/Datasets/ms_isbi2015/train_2D','img_'+file.split('.')[0]+'.nii'))
    nib.save(mask, os.path.join('/home/hao/Documents/Datasets/ms_isbi2015/train_2D','mask_'+file.split('.')[0]+'.nii'))


# for i in sorted(os.listdir('/home/hao/Documents/normalized_ISBI2015/train/')):
#     print(i)
#     pkltonii('/home/hao/Documents/normalized_ISBI2015/train/',i)