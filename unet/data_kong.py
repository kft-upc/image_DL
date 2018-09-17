# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 16:57:03 2018

@author: kft_Terra
"""

from __future__ import print_function

import os
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
import numpy as np


Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])



def trainGenerator(batch_size, aug_dict,train_path,image_folder,mask_folder,save_dir, target_size,
                    num_class = 2 ,image_color_mode = 'grayscale', mask_color_mode = 'grayscale', 
                    image_prefix='image', mask_prefix='mask', save_format='png',seed = 1):
    
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow_from_directory(
            train_path,
            target_size = target_size,
            classes = [image_folder],
            color_mode = image_color_mode,
            class_mode = None,
            batch_size= batch_size,
            save_to_dir= save_dir,
            save_format=save_format,
            save_prefix=image_prefix,
            seed = seed)
    
    mask_generator = mask_datagen.flow_from_directory(
            train_path,
            target_size=target_size,
            classes= [mask_folder],
            color_mode = mask_color_mode,
            class_mode = None,
            batch_size= batch_size,
            save_to_dir= save_dir,
            save_format=save_format,
            save_prefix=mask_prefix,
            seed = seed)
    
    train_generator = zip(image_generator, mask_generator)
    
    for img,mask in train_generator:
        img,mask = adjustData(img,mask)
        yield (img,mask)
    
    

def adjustData(img,mask):
    if(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)

def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        img = img / 255
        img = resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) 
        img = np.reshape(img,(1,)+img.shape)
        yield img   

def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255

def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)

        
#if __name__ == '__main__':
    #train_path = os.path.join(os.getcwd(),'far','train')
   # image_folder = 'image'
   # mask_folder = 'label'
  #  save_dir = os.path.join(train_path,'aug')
  #  target_size = (256,256)
  #  aug_dict = {}
  #  datagen = trainGenerator(32,aug_dict,train_path,image_folder,mask_folder,save_dir,target_size)
   # next(datagen)
    