import SimpleITK as sitk
import itk
from DicomRTTool.ReaderWriter import DicomReaderWriter, ROIAssociationClass
import os
from glob import glob
import numpy as np

#TODO: Save some kind of log over the progress

def GetROINames(rois):
    searchNames = ['CTV','PTV','Bladder','Rectum']
    foundNames = list()
    for nm in searchNames:
      for roi in rois:
         if nm.lower() in roi.lower():
            foundNames.append(roi)
    return foundNames

def GetBoundingBox(mask):
    mask = mask > 0
    flt = sitk.LabelShapeStatisticsImageFilter()
    flt.Execute(mask)
    return flt.GetBoundingBox(1)

def CropImage(im,bbox,margins=[0,0,0,0,0,0]):
    flt = sitk.ExtractImageFilter()
    indices = list()
    sizes = list()
    output = list()
    imSize = im[0].GetSize()
    for i in range(3):
        indices.append(bbox[i]+margins[i])
        if indices[i] < 0:
            indices[i] = 0
        sizes.append(bbox[i+3]+margins[i+3])
        if sizes[i] > imSize[i]:
            sizes[i] = imSize[i]
    flt.SetIndex(indices)
    flt.SetSize(sizes)
    for item in im:
        output.append(flt.Execute(item))
    return output

def ConvertSITKtoITK(sitk_image):
    image_dimension = sitk_image.GetDimension()
    itk_image = itk.GetImageFromArray(sitk.GetArrayFromImage(sitk_image))
    itk_image.SetOrigin(sitk_image.GetOrigin())
    itk_image.SetSpacing(sitk_image.GetSpacing())
    itk_image.SetDirection(
       itk.GetMatrixFromArray(np.reshape(np.array(sitk_image.GetDirection()), [image_dimension] * 2)))
    return itk_image

def ConvertITKtoSITK(itk_image):
    new_sitk_image = sitk.GetImageFromArray(itk.GetArrayFromImage(itk_image))
    new_sitk_image.SetOrigin(tuple(itk_image.GetOrigin()))
    new_sitk_image.SetSpacing(tuple(itk_image.GetSpacing()))
    new_sitk_image.SetDirection(itk.GetArrayFromMatrix(itk_image.GetDirection()).flatten())
    return new_sitk_image

rootData = 'P:/TERAPI/FYSIKER/David_Tilly/DosePrediction/PatientData'
patients = os.listdir(rootData)
cropMargin = [-30,-30,-30,50,50,50] #A bit strange giving margins like this (first three are starting indices, second sizes (in mm?) but don't care since close to 1mm isotropic resolution
#rootOutput = 'C:/Users/frs039/testIntrafractionProstate'
rootOutput = 'E:/prostateIntrafractionMotion'

if os.path.isfile(rootOutput+'/log.txt'):
    os.remove(rootOutput+'/log.txt')

for patient in patients:
   rootPatient = os.path.join(rootData,patient)
   rootOut = os.path.join(rootOutput,patient)
   #Planning images are named 'fx{n]', verification 'VER{n}' and post images 'POST{n}'
   planningImages = glob(rootPatient+'/fx*')
   for pIm in planningImages:
       try:
          fractionNr = pIm[-1] #Assuming last character is fraction number
          verificationPath = rootPatient+'/VER'+fractionNr
          postPath = rootPatient+'/POST'+fractionNr
          verificationPostPaths = dict()
          if os.path.isdir(verificationPath):
             verificationPostPaths['VER'] = verificationPath
          if os.path.isdir(postPath):
             verificationPostPaths['POST'] = postPath
          if len(verificationPostPaths.keys()) == 0:
             continue

          #Read planning image and contours
          Dicom_reader = DicomReaderWriter(description='Examples', arg_max=True)
          Dicom_reader.walk_through_folders(pIm)
          allRois = Dicom_reader.return_rois(print_rois=False)
          roiNames = GetROINames(allRois)
          # Seems like loading masks in sparse format, which is problematic due to overlapping structures.
          masks = list()
          for roi in roiNames:
             Dicom_reader.set_contour_names_and_associations(contour_names=[roi])
             Dicom_reader.get_images_and_mask()
             mask = Dicom_reader.annotation_handle
             masks.append(mask)
          #Extract full mask to enable cropping based on all structures
          Dicom_reader.set_contour_names_and_associations(contour_names=roiNames)
          Dicom_reader.get_images_and_mask()
          fullMask = Dicom_reader.annotation_handle
          bbox = GetBoundingBox(fullMask)
          #Extract image
          planningImage = Dicom_reader.dicom_handle
          planningImage = CropImage([planningImage],bbox,margins=cropMargin)[0]
          planningDateTime = [] #TODO
          #Crop mask as well
          masks = CropImage(masks, bbox, margins=cropMargin)
          #Save image and mask as .nii
          rootPlan = os.path.join(rootOut,fractionNr,'PLAN')
          if not os.path.isdir(rootPlan): os.makedirs(rootPlan)
          sitk.WriteImage(sitk.Cast(planningImage,sitk.sitkFloat32),rootPlan+'/image.nii.gz')
          for mask,roiName in zip(masks,roiNames):
             sitk.WriteImage(sitk.Cast(mask,sitk.sitkInt8),rootPlan+'/mask_'+roiName+'.nii.gz')

          #Read verification and post image
          for keyPath in verificationPostPaths.keys():
            Dicom_reader_ver = DicomReaderWriter(description='Examples', arg_max=True)
            Dicom_reader_ver.walk_through_folders(verificationPostPaths[keyPath])
            Dicom_reader_ver.get_images()
            image = Dicom_reader_ver.dicom_handle
            image = CropImage([image],bbox,margins=cropMargin)[0]
            #Perform registration
            parameter_object = itk.ParameterObject.New()
            default_bspline_parameter_map = parameter_object.GetDefaultParameterMap('bspline', 4)
            default_bspline_parameter_map['FinalBSplineInterpolationOrder'] = ['1']
            parameter_object.AddParameterMap(default_bspline_parameter_map)
            rootSave = os.path.join(rootOut, fractionNr, keyPath)
            if not os.path.isdir(rootSave): os.makedirs(rootSave)
            #Save verification/post image
            sitk.WriteImage(sitk.Cast(image, sitk.sitkFloat32), rootSave+'/image_original.nii.gz')
            result_image_bspline, result_transform_parameters = itk.elastix_registration_method(
               ConvertSITKtoITK(image), ConvertSITKtoITK(planningImage),
               parameter_object=parameter_object,
               log_to_console=True,
               output_directory = rootSave)
            #Save image
            #result_image_bspline = ConvertITKtoSITK(result_image_bspline)
            #sitk.WriteImage(sitk.Cast(result_image_bspline,sitk.sitkFloat32),rootSave+'/image_registered.nii.gz')

            #Apply transform to all masks
            for mask,roiName in zip(masks,roiNames):
                mask = sitk.Cast(mask,sitk.sitkFloat32)
                transformix_object = itk.TransformixFilter.New(ConvertSITKtoITK(mask))
                #Read transform parameters
                parameter_object = itk.ParameterObject.New()
                parameter_object.ReadParameterFile(rootSave+'/Transformparameters.0.txt')
                parameter_object.SetParameter('FinalBSplineInterpolationOrder', '0')
                transformix_object.SetTransformParameterObject(parameter_object)
                transformix_object.UpdateLargestPossibleRegion()
                result_mask = transformix_object.GetOutput()
                #Save mask
                result_mask = ConvertITKtoSITK(result_mask)
                sitk.WriteImage(sitk.Cast(result_mask,sitk.sitkInt8), rootSave + '/mask_'+roiName+'.nii.gz')
       except Exception as Argument:
          f = open(rootOutput+'/log.txt', 'a')
          f.write(str(Argument))
          f.close()




