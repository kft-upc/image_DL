# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 20:18:22 2018

@author: kft_Terra
"""

import os
import numpy as np

from skimage.io import imsave, imread

data_path = os.path.join(os.getcwd(),'far')

def train_image():
        
    train_label_path = os.path.join(data_path,'test/label')
    images = os.listdir(train_label_path)
    total = len(images)
    i = 0
    for image_name in images:         
        
        img_mask = imread(os.path.join(train_label_path, image_name), as_gray=True)          
        img_mask[img_mask == 0] = 1
        img_mask[img_mask == 255] = 0
        img_mask[img_mask == 1] = 255        

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
        imsave(os.path.join(train_label_path, image_name),img_mask)
    print('Loading done.')
    
if __name__ == '__main__':
    train_image()