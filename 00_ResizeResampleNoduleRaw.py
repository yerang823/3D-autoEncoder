# -*- conding: utf-8 -*-

#========= resize ==========#

import cv2
import numpy as np
import cv2
import numpy as np
import glob
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tqdm import tqdm
import zipfile
from tqdm import tnrange, tqdm_notebook

def ResizeNoduleRaw(path):
    
    file=path.split('/')[-1]
    size=file.split('.')[1]
    x=int(size.split('x')[0])
    y=int(size.split('x')[1])
    z=int(size.split('x')[2])

    with open(path, 'rb') as fid:
        data_array = np.fromfile(fid, np.int16).reshape((z,y,x))
    img_stack=np.array(data_array)

    width = 32
    height = 32
    vox = np.zeros((z, width, height))

    for idx in range(z):
        img = img_stack[idx, :, :]
        img_sm = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        vox[idx, :, :] = img_sm
        
    print('file=',file)
    
    return vox

def ResampleImage(itk_image, out_spacing, is_label=False):
    
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    out_size = [int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))), 
                int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
                32] #int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)
        
    return resample.Execute(itk_image)

def ResampleAndSave(voxel,SavePath):
    itk_img = sitk.GetImageFromArray(voxel.astype('int16'))
    itk_img_resam = ResampleImage(itk_img, out_spacing=[1, 1, round(itk_img.GetSize()[2]/32,2)], is_label=False)
    
    print(itk_img.GetSize(),itk_img.GetSpacing(),'--->',itk_img_resam.GetSize(),itk_img_resam.GetSpacing())

    array = sitk.GetArrayFromImage(itk_img_resam)
    
    x=itk_img_resam.GetSize()[0]
    y=itk_img_resam.GetSize()[1]
    z=itk_img_resam.GetSize()[2]
    
    f=open(SavePath[:-4]+"_%sx%sx%s.raw"%(str(x),str(y),str(z)),'wb')
    f.write(array.reshape(-1).tobytes())
    f.close()
    
    np.save(SavePath[:-4]+"_%sx%sx%s.npy"%(str(x),str(y),str(z)),array)
    
    print('Saved',SavePath)

    
    
# 32x32x32 save
path='../data/raw/'
save_path='../data/raw_32x32x32/'

li=os.listdir(path)

for i in range(len(li)):
    voxel=ResizeNoduleRaw(path+li[i])
    ResampleAndSave(voxel,save_path+li[i])
 

 
'''
# 32x32x32 new save
path=
save_path_32=

li=os.listdir(path)

for i in range(len(li)):
    x_size=int(li[i].split('.')[1].split('x')[0])
    if li[i][-4:]=='.npy' and x_size > 10: #if it is numpy and size over 10, crop it
        npy=np.load(path+li[i])
        npy_32=npy[16:48,16:48,16:48]
        name=li[i].split('_')[0]+'_'+li[i].split('_')[1]+'_32x32x32.npy'
        np.save(save_path_32+name,npy_32)
        print(npy[32:,32:,32:].shape, save_path_32+name)
        
        # save raw to check with imageJ
#         name_1=li[i].split('_')[0]+'_'+li[i].split('_')[1]+'_32x32x32.raw'
#         f=open( save_path_32+name_1,'wb')
#         f.write(npy_32.reshape(-1).tobytes())
#         f.close()
'''