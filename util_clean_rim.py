import os
import numpy as np
import nibabel as nib
import ants
import monai
from shutil import copyfile
import glob
root1 = '/home/hao/Documents/rim_resample'
root = '/home/hao/Documents/nnUNet_raw/nnUNet_raw_data/Task001_rim/imagesTr'
write = '/home/hao/Documents/rim_tr_images/test'
lable = '/home/hao/Documents/tmpp'
import SimpleITK as sitk
from typing import Tuple
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *


def get_identifiers_from_splitted_files(folder: str):
    uniques = np.unique([i[:-12] for i in subfiles(folder, suffix='.nii.gz', join=False)])
    return uniques


def generate_dataset_json(output_file: str, imagesTr_dir: str, imagesTs_dir: str, modalities: Tuple,
                          labels: dict, dataset_name: str, license: str = "hands off!", dataset_description: str = "",
                          dataset_reference="", dataset_release='0.0'):
    """
    :param output_file: This needs to be the full path to the dataset.json you intend to write, so
    output_file='DATASET_PATH/dataset.json' where the folder DATASET_PATH points to is the one with the
    imagesTr and labelsTr subfolders
    :param imagesTr_dir: path to the imagesTr folder of that dataset
    :param imagesTs_dir: path to the imagesTs folder of that dataset. Can be None
    :param modalities: tuple of strings with modality names. must be in the same order as the images (first entry
    corresponds to _0000.nii.gz, etc). Example: ('T1', 'T2', 'FLAIR').
    :param labels: dict with int->str (key->value) mapping the label IDs to label names. Note that 0 is always
    supposed to be background! Example: {0: 'background', 1: 'edema', 2: 'enhancing tumor'}
    :param dataset_name: The name of the dataset. Can be anything you want
    :param license:
    :param dataset_description:
    :param dataset_reference: website of the dataset, if available
    :param dataset_release:
    :return:
    """
    train_identifiers = get_identifiers_from_splitted_files(imagesTr_dir)

    if imagesTs_dir is not None:
        test_identifiers = get_identifiers_from_splitted_files(imagesTs_dir)
    else:
        test_identifiers = []

    json_dict = {}
    json_dict['name'] = dataset_name
    json_dict['description'] = dataset_description
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = dataset_reference
    json_dict['licence'] = license
    json_dict['release'] = dataset_release
    json_dict['modality'] = {str(i): modalities[i] for i in range(len(modalities))}
    json_dict['labels'] = {str(i): labels[i] for i in labels.keys()}

    json_dict['numTraining'] = len(train_identifiers)
    json_dict['numTest'] = len(test_identifiers)
    json_dict['training'] = [
        {'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i
        in
        train_identifiers]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_identifiers]

    if not output_file.endswith("dataset.json"):
        print("WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "
              "Proceeding anyways...")
    save_json(json_dict, os.path.join(output_file))

def flip():
    for file in os.listdir(lable):

        img = nib.load(os.path.join(lable, file))
        img_data = img.get_fdata()
        img_data1 = np.flip(img_data, axis=1)
        img1 = nib.Nifti1Image(img_data1, img.affine, img.header)
        nib.save(img1, os.path.join(write, file))

def n4():
    for file in os.listdir(root):
        img = ants.image_read(os.path.join(root, file))
        img = ants.n4_bias_field_correction(img, verbose=True)
        ants.image_write(img, os.path.join(root, file))
import scipy

def resize_data_volume_by_scale(data, scale):
   """
   Resize the data based on the provided scale
   """
   (x1,x2,x3) = scale
   scale_list = [x1,x2,x3]
   return scipy.ndimage.interpolation.zoom(data, scale_list, order=0)

def resample_volume(volume_path, interpolator = sitk.sitkLinear, new_spacing = [0.55, 0.55, 0.8]):
    volume = sitk.ReadImage(volume_path, sitk.sitkFloat32) # read and cast to float32
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(original_size, original_spacing, new_spacing)]
    return sitk.Resample(volume, new_size, sitk.Transform(), interpolator,
                         volume.GetOrigin(), new_spacing, volume.GetDirection(), 0,
                         volume.GetPixelID())

def normalize_zerone(data_path):
    image = nib.load(data_path)
    data = image.get_fdata()
    normalize = np.array([(x - np.amin(data)) / (np.amax(data) - np.amin(data)) for x in data])
    #normalize1 = np.array([(x - np.min(data)) / (np.max(data) - np.min(data)) for x in data])
    return nib.Nifti1Image(normalize, image.affine, image.header)

def changefilename():
    list = [i.split('_')[1] for i in sorted(os.listdir(root))]
    print(list)
    list = sorted([i for i in set(list)])
    print(list)
    dict = {x:'%03d' % y for y,x in enumerate(list)}
    print(dict)

    for i, file in enumerate(sorted(os.listdir(root))):
        name = file.split('_')[1]
        dst = file.replace(name,str(dict[name]))
        os.rename(os.path.join(root,file),os.path.join(root,dst))

if __name__ == '__main__':
    # generate_dataset_json(output_file='/home/hao/Documents/Task001_Rim/dataset.json',
    #                       imagesTr_dir='/home/hao/Documents/Task001_Rim/imagesTr',
    #                       imagesTs_dir=None,
    #                       modalities=('flair','swi','fswi'),
    #                       labels = {0:'background', 1:'rim'},
    #                       dataset_name='rim')

        # print(file)
        # name = file.split('_')[-1].split('.')[0]
        # if name == 'mask':
        #     print('A mask')
        #     continue
        # normalized = normalize_zerone(os.path.join(root,file))
        # nib.save(normalized, os.path.join(write,file))
    img = nib.load(os.path.join(root, 'rim_017_0002.nii.gz'))
    img_data = img.get_fdata()
    img_data1 = np.flip(img_data, axis=1)
    img1 = nib.Nifti1Image(img_data1, img.affine, img.header)
    nib.save(img1, os.path.join(root, 'rim_017_0002.nii.gz'))
        #img = nib.load(os.path.join(root,file))
        #image = resample_volume(os.path.join(root,file))
        #sitk.WriteImage(image, os.path.join(write,file))
        #data = img.get_fdata()
        #print(data.shape)

        #data1 = resize_data_volume_by_scale(data, scale = (1,1,2.4))
        #img1 = nib.Nifti1Image(data1, img.affine, img.header)
        #nib.save(img1, os.path.join(write,file))
        # img = ants.image_read(os.path.join(root, file))
        #
        # img = ants.pad_image(img, shape=(320,640,640))
        # ants.image_write(img, os.path.join(write,file))
    # name = []
    # for file in os.listdir(os.path.join(root,'rim_tr_labels')):
    #     name.append(file.split('_')[0])
    # print(name)
    # dst = os.path.join(root, 'rim_labels')
    # #for i in sorted(name):
    #     #print(i)
    # src = os.path.join(root, 'Rim_Data')
    # src += '/*binary_mask.nii*'
    # print(src)
    # for name in glob.glob(src, recursive=True):
    #     print(name)
    #     #copyfile(file, dst)
    #
    #
    # # name1 = []
    # for file in os.listdir(os.path.join(root, 'rim_label')):
    #     name1.append(file.split('_')[0])
    # name = set(name)
    # print(sorted(name))
    # print(sorted(name1))
#mi = ants.image_read('/home/hao/Documents/rim_datasets/339117_FLAIR.nii')
# #fi = ants.resample_image(fi, (60,60), 1, 0)
# #mi = ants.resample_image(mi, (60,60), 1, 0)
#
# mytx = ants.registration(fixed=fi, moving=mi, type_of_transform = 'SyN' )
# ants.image_write(mytx, '/hom/hao/Documents/test.nii')