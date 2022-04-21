import os
import nibabel as nib
import glob
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

scaler = MinMaxScaler()

def load_volume(base_directory, image_num, channel, training=True):

    image_num = str(image_num)
    if channel in ["t1", "t2", "t1ce", "flair", "seg"]:
        channel = str(channel)
    else:
        print("Error: Not valid channel")
        return None

    if training==True:
        type_path = "Training"
    else:
        type_path = "Validation"

    return nib.load(base_directory+"BraTS20_"+type_path+"_"+image_num +"/BraTS20_"+type_path+"_"+image_num+"_"+channel+".nii").get_fdata()


def get_channel_lists(base_directory, channel=None, test_data=False):
    
    if channel is not None:
        channel_list = sorted(glob.glob(base_directory+"*/*"+str(channel)+".nii"))
        return channel_list
    else:
        t2_list = sorted(glob.glob(base_directory+"*/*t2.nii"))
        t1ce_list = sorted(glob.glob(base_directory+"*/*t1ce.nii"))
        flair_list = sorted(glob.glob(base_directory+"*/*flair.nii"))
        if test_data:
            return t1ce_list, t2_list, flair_list
        else:
            mask_list = sorted(glob.glob(base_directory+"*/*seg.nii"))
            return t1ce_list, t2_list, flair_list, mask_list


def plot_random_volume(base_directory, image_num=None, slice=None):

    if image_num is None:
        n_image = get_image_number()
    else:
        n_image = str(image_num)

    t1_volume = load_volume(base_directory, n_image, "t1")
    t1ce_volume = load_volume(base_directory, n_image, "t1ce")
    t2_volume = load_volume(base_directory, n_image, "t2")
    flair_volume = load_volume(base_directory, n_image, "flair")
    mask = load_volume(base_directory, n_image, "seg")

    t1_volume=scaler.fit_transform(t1_volume.reshape(-1, t1_volume.shape[-1])).reshape(t1_volume.shape)
    t1ce_volume=scaler.fit_transform(t1ce_volume.reshape(-1, t1ce_volume.shape[-1])).reshape(t1ce_volume.shape)
    t2_volume=scaler.fit_transform(t2_volume.reshape(-1, t2_volume.shape[-1])).reshape(t2_volume.shape)
    flair_volume=scaler.fit_transform(flair_volume.reshape(-1, flair_volume.shape[-1])).reshape(flair_volume.shape)

    mask[mask==4] = 3

    if slice is None:
        n_slice = random.randint(0, mask.shape[2])
    else:
        n_slice = int(slice)

    # Print the images
    plt.figure(figsize = (12,8))
    plt.subplot(231)
    plt.imshow(flair_volume[:,:,n_slice], cmap='gray')
    plt.title('Test Image: flair')
    plt.subplot(232)
    plt.imshow(t1_volume[:,:,n_slice], cmap='gray')
    plt.title('Test Image: t1')
    plt.subplot(233)
    plt.imshow(t1ce_volume[:,:,n_slice], cmap='gray')
    plt.title('Test Image: t1ce')
    plt.subplot(234)
    plt.imshow(t2_volume[:,:,n_slice], cmap='gray')
    plt.title('Test Image: t2')
    plt.subplot(235)
    plt.imshow(mask[:,:,n_slice])
    plt.title('Test Mask')
    plt.suptitle(f"3D volume slice: {n_slice}")
    plt.show()


def get_image_number(training_image=True):
    if training_image:
        image_number = np.random.randint(1, 370)
    else:
        image_number = np.random.randint(1, 125)

    if image_number < 100:
        if image_number < 10:
            image_number = "00" + str(image_number)
        else:
            image_number = "0" + str(image_number)
    else:
        image_number = str(image_number) 

    return image_number

def get_dimensions(base_directory):
    """Prints the dimensions of a random training volume and it's mask
    """

    image_number = get_image_number()              
    
    test_image = nib.load(base_directory + 'BraTS20_Training_' + image_number + '/BraTS20_Training_' + image_number + '_flair.nii').get_fdata()
    test_mask = nib.load(base_directory + 'BraTS20_Training_' + image_number + '/BraTS20_Training_' + image_number + '_seg.nii').get_fdata()

    print(f"Dimensions of the 3D volume is: {test_image.shape}")
    print(f"Dimensions of the 3D volume mask is: {test_mask.shape}")
    return None

def save_training_volume_to_numpy(t2_list, t1ce_list, flair_list, mask_list, train_img_path, train_mask_path):
    # Load, MinMax scale, combine and crop all training images then save
    for img in range(len(t2_list)):        
        temp_image_t2=nib.load(t2_list[img]).get_fdata()
        temp_image_t2=scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)
    
        temp_image_t1ce=nib.load(t1ce_list[img]).get_fdata()
        temp_image_t1ce=scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)
    
        temp_image_flair=nib.load(flair_list[img]).get_fdata()
        temp_image_flair=scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)
            
        temp_mask=nib.load(mask_list[img]).get_fdata()
        temp_mask=temp_mask.astype(np.uint8)
        temp_mask[temp_mask==4] = 3  #Reassign mask values 4 to 3
        
        temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)
        
        #Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches. 
        #cropping x, y, and z
        temp_combined_images = temp_combined_images[56:184, 56:184, 13:141]
        temp_mask = temp_mask[56:184, 56:184, 13:141]
        
        val, counts = np.unique(temp_mask, return_counts=True)
        
        if (1 - (counts[0]/counts.sum())) > 0.01:  #At least 1% useful volume with labels that are not 0
            temp_mask= to_categorical(temp_mask, num_classes=4)
            np.save(train_img_path + '/image_' +str(img) + '.npy', temp_combined_images)
            np.save(train_mask_path + '/mask_'+str(img)+'.npy', temp_mask)

def save_test_volume_to_numpy(t2_list, t1ce_list, flair_list, test_img_path):
    # Load, MinMax scale, combine and crop all test images (which we will use for testing) then save
    for img in range(len(t2_list)):
        #print("Now preparing image and masks number: ", img)
        
        temp_image_t2=nib.load(t2_list[img]).get_fdata()
        temp_image_t2=scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)
    
        temp_image_t1ce=nib.load(t1ce_list[img]).get_fdata()
        temp_image_t1ce=scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)
    
        temp_image_flair=nib.load(flair_list[img]).get_fdata()
        temp_image_flair=scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)
        
        temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)
        
        #Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches. 
        #cropping x, y, and z
        temp_combined_images = temp_combined_images[56:184, 56:184, 13:141]

        np.save(test_img_path + '/testimage_' + str(img) + '.npy', temp_combined_images)
