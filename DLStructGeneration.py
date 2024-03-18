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
    [fromIm.SetMetaData(key,toIm.GetMetaData(key)) for key in toIm.GetMetaDataKeys()]

    return toIm

pathModels = 'C:/Users/frs039/OneDrive - Region Uppsala/Paper IV/SegmentationModels'
pathImages = 'E:/prostateIntrafractionMotion'

modelNames = os.listdir(pathModels)
activation = tf.keras.layers.LeakyReLU(1e-2)

models = list()
for nm in modelNames:
    models.append(tf.keras.models.load_model(os.path.join(pathModels,nm),custom_objects={'LeakyReLU':activation}))

patients = os.listdir(pathImages)

for pat in patients[0]:
    patRoot = os.path.join(pathImages,pat)
    fractions = os.listdir(patRoot)
    for fr in fractions[0]:
        frRoot = os.path.join(patRoot,fr)
        imagePaths = os.listdir(frRoot)
        for imPath in imagePaths:
            image = os.path.join(frRoot,imPath,'image.nii.gz')
            if not os.path.isfile(image):
                image = os.path.join(frRoot,imPath,'image_original.nii.gz')
            im = sitk.ReadImage(image)
            imArray = sitk.GetArrayFromImage(im)
            imArray = z_normalize(imArray)
            segmentation = 0
            imArray = np.expand_dims(imArray,axis=0)
            imArray = np.expand_dims(imArray,axis=-1)
            for model in models:
                tmp = patchInference(model, imArray, patchSize=[96,96,96],stride=[48,48,48])
                segmentation += tmp[0]
            segmentation = segmentation / len(models)
            segmentation = max_along_axis(segmentation)
            bladder = segmentation[0,...,2]
            rectum = segmentation[0,...,3]
            bladderSitk = sitk.GetImageFromArray(bladder)
            rectumSitk = sitk.GetImageFromArray(rectum)
            bladderSitk = SetMetadata(imArray, bladderSitk)
            rectumSitk = SetMetadata(imArray, rectumSitk)
            #Save
            sitk.WriteImage(bladderSitk,frRoot+'/bladder_DL.nii.gz')
            sitk.WriteImage(rectumSitk,frRoot+'/rectum_DL.nii.gz')

            
            
            

            
            
     
 
 