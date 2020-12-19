#KIDNEY DATA PREP

# %cd /notebooks 
import SimpleITK as sitk
import numpy as np

for i in range(20):
    image=sitk.GetArrayFromImage(sitk.ReadImage('AMS_izziv/ams_izziv/training_data_v2/kidney/Training/case'+str(i+1).zfill(2)+'/image.nii.gz'))
    image = (255*(image-np.min(image))/np.max(image-np.min(image))).astype(np.uint8)
    image = np.pad(image, ((512-image.shape[0], 0),(512-image.shape[0], 0)), 'constant', constant_values = 0)
    image[:,256:] = np.zeros((512,256))
    sitk.WriteImage(sitk.GetImageFromArray(image),'AMS_izziv/pytorch-unet-segmentation-master/data/train/images/'+str((i*3))+'.png')
    sitk.WriteImage(sitk.GetImageFromArray(image),'AMS_izziv/pytorch-unet-segmentation-master/data/train/images/'+str((i*3)+1)+'.png')
    sitk.WriteImage(sitk.GetImageFromArray(image),'AMS_izziv/pytorch-unet-segmentation-master/data/train/images/'+str((i*3)+2)+'.png')
    
    image=sitk.GetArrayFromImage(sitk.ReadImage('AMS_izziv/ams_izziv/training_data_v2/kidney/Training/case'+str(i+1).zfill(2)+'/task01_seg01.nii.gz'))
    image = (255*(image-np.min(image))/np.max(image-np.min(image))).astype(np.uint8)
    image = np.pad(image, ((512-image.shape[0], 0),(512-image.shape[0], 0)), 'constant', constant_values = 0)
    sitk.WriteImage(sitk.GetImageFromArray(image),'AMS_izziv/pytorch-unet-segmentation-master/data/train/masks/'+str((i*3))+'.png')
    
    image=sitk.GetArrayFromImage(sitk.ReadImage('AMS_izziv/ams_izziv/training_data_v2/kidney/Training/case'+str(i+1).zfill(2)+'/task01_seg02.nii.gz'))
    image = (255*(image-np.min(image))/np.max(image-np.min(image))).astype(np.uint8)
    image = np.pad(image, ((512-image.shape[0], 0),(512-image.shape[0], 0)), 'constant', constant_values = 0)
    sitk.WriteImage(sitk.GetImageFromArray(image),'AMS_izziv/pytorch-unet-segmentation-master/data/train/masks/'+str((i*3)+1)+'.png')
    
    image=sitk.GetArrayFromImage(sitk.ReadImage('AMS_izziv/ams_izziv/training_data_v2/kidney/Training/case'+str(i+1).zfill(2)+'/task01_seg03.nii.gz'))
    image = (255*(image-np.min(image))/np.max(image-np.min(image))).astype(np.uint8)
    image = np.pad(image, ((512-image.shape[0], 0),(512-image.shape[0], 0)), 'constant', constant_values = 0)
    sitk.WriteImage(sitk.GetImageFromArray(image),'AMS_izziv/pytorch-unet-segmentation-master/data/train/masks/'+str((i*3)+2)+'.png')
    
    


for i in range(4):
    image=sitk.GetArrayFromImage(sitk.ReadImage('AMS_izziv/ams_izziv/validation_data_v2/kidney/Validation/case'+str(i+21).zfill(2)+'/image.nii.gz'))
    image = (255*(image-np.min(image))/np.max(image-np.min(image))).astype(np.uint8)
    image = np.pad(image, ((512-image.shape[0], 0),(512-image.shape[0], 0)), 'constant', constant_values = 0)
    image[:,256:] = np.zeros((512,256))
    sitk.WriteImage(sitk.GetImageFromArray(image),'AMS_izziv/pytorch-unet-segmentation-master/data/val/images/'+str(i)+'.png')
    
    image1=sitk.GetArrayFromImage(sitk.ReadImage('AMS_izziv/ams_izziv/validation_data_v2/kidney/Validation/case'+str(i+21).zfill(2)+'/task01_seg01.nii.gz'))
    image1 = (255*(image1-np.min(image1))/np.max(image1-np.min(image1))).astype(np.float32)
    image1 = np.pad(image1, ((512-image1.shape[0], 0),(512-image1.shape[0], 0)), 'constant', constant_values = 0)

    image2=sitk.GetArrayFromImage(sitk.ReadImage('AMS_izziv/ams_izziv/validation_data_v2/kidney/Validation/case'+str(i+21).zfill(2)+'/task01_seg02.nii.gz'))
    image2 = (255*(image2-np.min(image2))/np.max(image2-np.min(image2))).astype(np.float32)
    image2 = np.pad(image2, ((512-image2.shape[0], 0),(512-image2.shape[0], 0)), 'constant', constant_values = 0)

    image3=sitk.GetArrayFromImage(sitk.ReadImage('AMS_izziv/ams_izziv/validation_data_v2/kidney/Validation/case'+str(i+21).zfill(2)+'/task01_seg03.nii.gz'))
    image3 = (255*(image3-np.min(image3))/np.max(image3-np.min(image3))).astype(np.float32)
    image3 = np.pad(image3, ((512-image3.shape[0], 0),(512-image3.shape[0], 0)), 'constant', constant_values = 0)
    img_avg = ((image1+image2+image3)/3.0).astype(np.uint8)
    sitk.WriteImage(sitk.GetImageFromArray(img_avg),'AMS_izziv/pytorch-unet-segmentation-master/data/val/masks/'+str(i)+'.png')
    
    
    
#BRAIN GROWTH DATA PREP

import SimpleITK as sitk
import numpy as np
task = 1
for i in range(34):
    for j in range(7):
        image=sitk.GetArrayFromImage(sitk.ReadImage('AMS_izziv/ams_izziv/training_data_v2/brain-growth/Training/case'+str(i+1).zfill(2)+'/image.nii.gz'))
        image = (255*(image-np.min(image))/np.max(image-np.min(image))).astype(np.uint8)
        image = np.pad(image, (((512-image.shape[0])//2, (512-image.shape[0])//2),((512-image.shape[0])//2, (512-image.shape[0])//2)), 'constant', constant_values = 0)
        sitk.WriteImage(sitk.GetImageFromArray(image),'AMS_izziv/pytorch-unet-segmentation-master/data/train/images/'+str(i*7+j)+'.png')
    
        mask=sitk.GetArrayFromImage(sitk.ReadImage('AMS_izziv/ams_izziv/training_data_v2/brain-growth/Training/case'+str(i+1).zfill(2)+'/task0'+str(task)+'_seg0'+str(j+1)+'.nii.gz'))
        mask = (255*(mask-np.min(mask))/np.max(mask-np.min(mask))).astype(np.uint8)
        mask = np.pad(mask, (((512-mask.shape[0])//2, (512-mask.shape[0])//2),((512-mask.shape[0])//2, (512-mask.shape[0])//2)), 'constant', constant_values = 0)
        sitk.WriteImage(sitk.GetImageFromArray(mask),'AMS_izziv/pytorch-unet-segmentation-master/data/train/masks/'+str(i*7+j)+'.png')

for i in range(5):
    for j in range(7):
        image=sitk.GetArrayFromImage(sitk.ReadImage('AMS_izziv/ams_izziv/validation_data_v2/brain-growth/Validation/case'+str(i+35).zfill(2)+'/image.nii.gz'))
        image = (255*(image-np.min(image))/np.max(image-np.min(image))).astype(np.uint8)
        image = np.pad(image, (((512-image.shape[0])//2, (512-image.shape[0])//2),((512-image.shape[0])//2, (512-image.shape[0])//2)), 'constant', constant_values = 0)
        sitk.WriteImage(sitk.GetImageFromArray(image),'AMS_izziv/pytorch-unet-segmentation-master/data/val/images/'+str(i*7+j)+'.png')
    
        mask=sitk.GetArrayFromImage(sitk.ReadImage('AMS_izziv/ams_izziv/validation_data_v2/brain-growth/Validation/case'+str(i+35).zfill(2)+'/task0'+str(task)+'_seg0'+str(j+1)+'.nii.gz'))
        mask = (255*(mask-np.min(mask))/np.max(mask-np.min(mask))).astype(np.uint8)
        mask = np.pad(mask, (((512-mask.shape[0])//2, (512-mask.shape[0])//2),((512-mask.shape[0])//2, (512-mask.shape[0])//2)), 'constant', constant_values = 0)
        sitk.WriteImage(sitk.GetImageFromArray(mask),'AMS_izziv/pytorch-unet-segmentation-master/data/val/masks/'+str(i*7+j)+'.png')
        
        
        
#PROSTATE TASK 1 DATA PREP
import SimpleITK as sitk
import numpy as np
import skimage
from skimage import transform
#parametri
num_seg = 6
train_cases = 48
val_cases = 7
#objekt
objekt = 'prostate'
task = 2
for i in range(train_cases):
    for j in range(num_seg):
        image=sitk.GetArrayFromImage(sitk.ReadImage('AMS_izziv/ams_izziv/training_data_v2/'+objekt+'/Training/case'+str(i+1).zfill(2)+'/image.nii.gz')).squeeze()
        image = (255*(image-np.min(image))/np.max(image-np.min(image))).astype(np.uint8)
        min_dim = np.min(image.shape)
        image_crop = image[0:min_dim,(image.shape[1]-min_dim)//2:(image.shape[1]+min_dim)//2]
        image_crop = skimage.transform.resize(image_crop, (512,512), order=None, mode='reflect', cval=0, clip=True, preserve_range=True, anti_aliasing=None, anti_aliasing_sigma=None).astype(np.uint8)
        sitk.WriteImage(sitk.GetImageFromArray(image_crop),'AMS_izziv/pytorch-unet-segmentation-master/data/train/images/'+str(i*num_seg+j)+'.png')
    
        mask=sitk.GetArrayFromImage(sitk.ReadImage('AMS_izziv/ams_izziv/training_data_v2/'+objekt+'/Training/case'+str(i+1).zfill(2)+'/task0'+str(task)+'_seg0'+str(j+1)+'.nii.gz')).squeeze()
        mask = (255*(mask-np.min(mask))/np.max(mask-np.min(mask))).astype(np.uint8)
        min_dim = np.min(mask.shape)
        mask_crop = mask[0:min_dim,(mask.shape[1]-min_dim)//2:(mask.shape[1]+min_dim)//2]
        mask_crop = skimage.transform.resize(mask_crop, (512,512), order=None, mode='reflect', cval=0, clip=True, preserve_range=True, anti_aliasing=None, anti_aliasing_sigma=None).astype(np.uint8)
        sitk.WriteImage(sitk.GetImageFromArray(mask_crop),'AMS_izziv/pytorch-unet-segmentation-master/data/train/masks/'+str(i*num_seg+j)+'.png')

        

for i in range(val_cases):

    for j in range(num_seg):
        image = sitk.GetArrayFromImage(sitk.ReadImage('AMS_izziv/ams_izziv/validation_data_v2/'+objekt+'/Validation/case'+str(i+train_cases+1).zfill(2)+'/image.nii.gz')).squeeze()
        image = (255*(image-np.min(image))/np.max(image-np.min(image))).astype(np.uint8)
        min_dim = np.min(image.shape)
        image_crop = image[0:min_dim,(image.shape[1]-min_dim)//2:(image.shape[1]+min_dim)//2]
        image_crop = skimage.transform.resize(image_crop, (512,512), order=None, mode='reflect', cval=0, clip=True, preserve_range=True, anti_aliasing=None, anti_aliasing_sigma=None).astype(np.uint8)
        sitk.WriteImage(sitk.GetImageFromArray(image_crop),'AMS_izziv/pytorch-unet-segmentation-master/data/val/images/'+str(i*num_seg+j)+'.png')
        
        mask = sitk.GetArrayFromImage(sitk.ReadImage('AMS_izziv/ams_izziv/validation_data_v2/'+objekt+'/Validation/case'+str(i+train_cases+1).zfill(2)+'/task0'+str(task)+'_seg0'+str(j+1)+'.nii.gz')).squeeze()
        mask = (255*(mask-np.min(mask))/np.max(mask-np.min(mask))).astype(np.uint8)
        min_dim = np.min(mask.shape)
        mask_crop = mask[0:min_dim,(mask.shape[1]-min_dim)//2:(mask.shape[1]+min_dim)//2]
        mask_crop = skimage.transform.resize(mask_crop, (512,512), order=None, mode='reflect', cval=0, clip=True, preserve_range=True, anti_aliasing=None, anti_aliasing_sigma=None).astype(np.uint8)
        sitk.WriteImage(sitk.GetImageFromArray(mask_crop),'AMS_izziv/pytorch-unet-segmentation-master/data/val/masks/'+str(i*num_seg+j)+'.png')
        
        
        

#BRAIN TUMOR TASK 1,2,3 DATA PREP
import SimpleITK as sitk
import numpy as np
#parametri
num_seg = 3
train_cases = 28
val_cases = 4
num_mod = 4
#objekt
objekt = 'brain-tumor'
task = 3
idx = 0
for i in range(train_cases):
    for j in range(num_seg):
        for k in range(num_mod):
            image=sitk.GetArrayFromImage(sitk.ReadImage('AMS_izziv/ams_izziv/training_data_v2/'+objekt+'/Training/case'+str(i+1).zfill(2)+'/image.nii.gz'))[k,:,:].squeeze()
            image = (255*(image-np.min(image))/np.max(image-np.min(image))).astype(np.uint8)
            image = np.pad(image, (((512-image.shape[0])//2, (512-image.shape[0])//2),((512-image.shape[1])//2, (512-image.shape[1])//2)), 'constant', constant_values = 0)
            sitk.WriteImage(sitk.GetImageFromArray(image),'AMS_izziv/pytorch-unet-segmentation-master/data/train/images/'+str(idx)+'.png')

            mask=sitk.GetArrayFromImage(sitk.ReadImage('AMS_izziv/ams_izziv/training_data_v2/'+objekt+'/Training/case'+str(i+1).zfill(2)+'/task0'+str(task)+'_seg0'+str(j+1)+'.nii.gz')).squeeze()
            mask = (255*(mask-np.min(mask))/np.max(mask-np.min(mask))).astype(np.uint8)
            mask = np.pad(mask, (((512-mask.shape[0])//2, (512-mask.shape[0])//2),((512-mask.shape[1])//2, (512-mask.shape[1])//2)), 'constant', constant_values = 0)
            sitk.WriteImage(sitk.GetImageFromArray(mask),'AMS_izziv/pytorch-unet-segmentation-master/data/train/masks/'+str(idx)+'.png')
            idx += 1
        
idx = 0
for i in range(val_cases):
    for j in range(num_seg):
        for k in range(num_mod):        
            image = sitk.GetArrayFromImage(sitk.ReadImage('AMS_izziv/ams_izziv/validation_data_v2/'+objekt+'/Validation/case'+str(i+train_cases+1).zfill(2)+'/image.nii.gz'))[k,:,:].squeeze()
            image = (255*(image-np.min(image))/np.max(image-np.min(image))).astype(np.uint8)
            image = np.pad(image, (((512-image.shape[0])//2, (512-image.shape[0])//2),((512-image.shape[0])//2, (512-image.shape[0])//2)), 'constant', constant_values = 0)
            sitk.WriteImage(sitk.GetImageFromArray(image),'AMS_izziv/pytorch-unet-segmentation-master/data/val/images/'+str(idx)+'.png')

            mask = sitk.GetArrayFromImage(sitk.ReadImage('AMS_izziv/ams_izziv/validation_data_v2/'+objekt+'/Validation/case'+str(i+train_cases+1).zfill(2)+'/task0'+str(task)+'_seg0'+str(j+1)+'.nii.gz')).squeeze()
            mask = (255*(mask-np.min(mask))/np.max(mask-np.min(mask))).astype(np.uint8)
            mask = np.pad(mask, (((512-mask.shape[0])//2, (512-mask.shape[0])//2),((512-mask.shape[1])//2, (512-mask.shape[1])//2)), 'constant', constant_values = 0)
            sitk.WriteImage(sitk.GetImageFromArray(mask),'AMS_izziv/pytorch-unet-segmentation-master/data/val/masks/'+str(idx)+'.png')