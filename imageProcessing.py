import numpy as np
from copy import deepcopy

def z_normalize(image, percentage_range = [0,100]):
    'Z-normalization of image'
    'image: input image'
    'percentage_range: intensity cropping outside defined range. Optional.'
    image = image.astype(np.float32)
    intensity_range = image.max() - image.min()
    intensity_max = image.max() - intensity_range * (1 - percentage_range[1] / 100)
    intensity_min = image.min() + intensity_range * percentage_range[0] / 100
    
    tmp_im = deepcopy(image)
    tmp_im = tmp_im[tmp_im < intensity_max]
    tmp_im = tmp_im[tmp_im > intensity_min]
    # breakpoint()
    mean = np.mean(tmp_im)
    std = np.std(tmp_im)
    
    return (image - mean) / std


def patchInference(model, im, patchSize, stride=[1,1,1], 
                   windowing = True, softmax_out = True):
    "Patch inference of model. The models is applied on a patches of" 
    "the image with a predefined stride and then concatenated into a"
    "single prediction"
    "Input: modelPath - path to tensorflow model"
    "       im - image size [n,x,y,(z)] where n is number of images"
    "       patchSize - patch size"
    "       stride - stride in patch inference"
    "       MC - Boolean wether to appy Monte Carlo sampling in inference. Default false"
    "       windowing - Boolean, wether to apply a hanning window in inference. Default true"
    "       softmax_out - Boolean, wether the output from the network is softmax"
    "       has implications on how to deal with overlapping patches and windowing"            
    "Output: Concatenated segmentations"
    
    imSiz = im.shape
    is3D = False
    if len(imSiz) > 3: is3D = True
    if is3D and len(imSiz) == 4: 
        im = np.expand_dims(im,axis=-1)
    
    maxX = imSiz[1] - patchSize[0]
    maxY = imSiz[2] - patchSize[1]
    if is3D: maxZ = imSiz[3] - patchSize[2]
    
    "Make a forward pass to determine the number of outputs"
    if not is3D: outTmp = model.predict(im[:,0:patchSize[0], 0:patchSize[1]])
    if is3D: outTmp = model.predict(im[:,0:patchSize[0],0:patchSize[1],0:patchSize[2],:])

    out_shape = list(outTmp.shape)
    "Since patch inference, extending the spatial component of the dimension"
    out_shape[1:-1] = im.shape[1:-1]
    comb = np.zeros(out_shape)
    
    "Need to keep track of the nr_predictions if not softmax output"
    if not softmax_out:
        nr_predictions = np.zeros(out_shape)
    
    "If MC inference also include the outputs for uncertainty"
    comb = [comb]
    
    xVals = np.arange(0,maxX,stride[0])
    xVals = np.append(xVals,maxX)
    yVals = np.arange(0,maxY,stride[1]) 
    yVals = np.append(yVals,maxY)

    if is3D: 
        zVals = np.arange(0,maxZ,stride[2]) 
        zVals = np.append(zVals,maxZ) 
    else:
        zVals = [None]
    print('x-axis starting indices are: '+str(xVals))
    print('y-axis starting indices are: '+str(yVals))
    if is3D: print('z-axis starting indices are: '+str(zVals))
    
    if windowing:
        w_x = np.hanning(patchSize[0])
        w_y = np.hanning(patchSize[1])
        w = np.outer(w_x,w_y)
        if is3D:
            w_z = np.hanning(patchSize[2])
            w = np.outer(w_z,np.ravel(w))
            w = np.reshape(w,patchSize)
    else:
        w = np.ones(patchSize)
        
    for x in xVals:
        for y in yVals:
            for z in zVals:
                x_patch = slice(x,patchSize[0]+x)
                y_patch = slice(y,patchSize[1]+y)
                patch = im[:,x_patch,y_patch,...]
                if z is not None:
                    z_patch = slice(z,patchSize[2]+z)
                patch = im[:,x_patch,y_patch,z_patch,...]
                out = model.predict(patch)
                out = list(out)
                for i,item in enumerate(out):
                    if z is None:
                        comb[i][:,x_patch,y_patch,:] = comb[i][:,x_patch,y_patch,:] + item * w[...,np.newaxis]
                        if not softmax_out:
                            nr_predictions[:,x_patch,y_patch,:] += w[...,np.newaxis]
                    if z is not None:
                        comb[i][:,x_patch,y_patch,z_patch,:] = comb[i][:,x_patch,y_patch,z_patch,:] + item * w[...,np.newaxis]
                        if not softmax_out:
                            nr_predictions[:,x_patch,y_patch,z_patch,:] += w[...,np.newaxis]
                        print(x,y,z)
                    else: print(x,y)
                    
    if softmax_out:
        "Need to re-normalize along the last axis due to overlapping patches"
        norm_val = np.sum(comb[i],axis=-1)
        comb[i] = comb[i] / norm_val[...,np.newaxis]
    else:
        comb[i] /= nr_predictions
   
    return comb

def max_along_axis(mask,axis=-1):
    "Assign the voxel the largest value along axis of an array"
    "Suitable e.g. with segmentation and a softmax output"
    ind = mask.argmax(axis=axis)
    n_dims = np.max(ind) + 1
    return np.eye(n_dims)[ind]