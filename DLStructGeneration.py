import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import os
import SimpleITK as sitk
from imageProcessing import z_normalize, patchInference, max_along_axis
 
def SetMetadata(fromIm,toIm):
    toIm.SetDirection(fromIm.GetDirection())
    toIm.SetOrigin(fromIm.GetOrigin())
    toIm.SetMetadata(fromIm.GetMetadata())
    return toIm

pathModels = 'C:/Users/frs039/OneDrive - Region Uppsala/Paper IV/SegmentationModels'
pathImages = 'E:/prostateIntrafractionMotion'

modelNames = os.listdir(pathModels)
activation = tf.keras.layers.LeakyReLU(1e-2)

models = list()
for nm in modelNames:
    models.append(tf.keras.models.load_model(nm,custom_objects={'LeakyReLU':activation}))

patients = os.listdir(pathImages)

for pat in patients[0]:
    patRoot = os.path.join(pathImages,pat)
    fractions = os.listdir(patRoot)
    for fr in fractions[0]:
        frRoot = os.path.join(patRoot,fr)
        imagePaths = os.listdir(frRoot)
        for imPath in imagePaths:
            im = sitk.ReadImage(imPath+'/image.nii.gz')
            imArray = sitk.GetArrayFromImage(im)
            imArray = z_normalize(imArray)
            segmentation = 0
            for model in models:
                tmp = patchInference(model, imArray, patchSize=[96,96,96],stride=[48,48,48])
                segmentation += tmp
            segmentation = segmentation / len(models)
            segmentation = max_along_axis(segmentation)
            bladder = segmentation[...,2]
            rectum = segmentation[...,3]
            bladderSitk = sitk.GetImageFromArray(bladder)
            rectumSitk = sitk.GetImageFromArray(rectum)
            bladderSitk = SetMetadata(imArray, bladderSitk)
            rectumSitk = SetMetadata(imArray, rectumSitk)
            #Save
            sitk.WriteImage(frRoot+'/bladder_DL.nii.gz',bladderSitk)
            sitk.WriteImage(frRoot+'/rectum_DL.nii.gz',rectumSitk)

            
            
            

            
            
     
 
 
