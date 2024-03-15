import numpy as np
from plotting import plot3D
import os
import SimpleITK as sitk
from glob import glob
import time

def ReadMasks(root) -> object:
    mask_paths = glob(root + '/mask*')
    masks = 0
    for p in mask_paths:
        if 'ptv' in p.lower():
            continue
        mask = sitk.ReadImage(p)
        masks += sitk.GetArrayFromImage(mask)
    return masks

root = 'E:/prostateIntrafractionMotion/retries'

pat = '11'
fr = '3'

patients = os.listdir(root)
imNames = ['PLAN', 'VER', 'POST']
#for pat in patients[0]:
patRoot = os.path.join(root, pat)
fractions = os.listdir(patRoot)
   # for fr in fractions[0:2]:
frRoot = os.path.join(patRoot, fr)
imageArray = sitk.ReadImage(frRoot + '/PLAN/image.nii.gz')
imageArray = sitk.GetArrayFromImage(imageArray)
masksArray = ReadMasks(frRoot+'/PLAN')
imageArray = imageArray / imageArray.max() + masksArray * 0.3
imageArray = np.expand_dims(imageArray, axis=-1)
verPostPaths = [frRoot + '/VER', frRoot + '/POST']
print('Patient ' + pat + ', Fraction ' + fr)
for vpPath in verPostPaths:
    if os.path.isdir(vpPath):
        im_original = sitk.ReadImage(vpPath + '/image_original.nii.gz')
        im_original = sitk.GetArrayFromImage(im_original)
        masksRegistered = ReadMasks(vpPath)
        im_original = im_original/im_original.max() + masksRegistered * 0.3
        im_original = np.expand_dims(im_original, axis=-1)
        imageArray = np.concatenate([imageArray, im_original], axis=-1)
tr = plot3D(imageArray)


