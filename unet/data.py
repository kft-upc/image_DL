from __future__ import print_function

import os
import numpy as np
from skimage.io import imsave, imread
from skimage.transform import resize

data_path = '/home/terra/ML/data/fardetect/'

image_rows = 256
image_cols = 256

   

        

def create_train_far_data():
    train_data_path = os.path.join(data_path,'train/image')
    train_label_path = os.path.join(data_path,'train/label')
    images = os.listdir(train_data_path)
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
         
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)
        img_mask = imread(os.path.join(train_label_path, image_name), as_grey=True)   

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(os.path.join(data_path,'train','imgs_train.npy'), imgs)
    np.save(os.path.join(data_path,'train','imgs_mask_train.npy'), imgs_mask)
    print('Saving to .npy files done.')

def create_train_data():
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    total = len(images) // 2

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)
        img_mask = imread(os.path.join(train_data_path, image_mask_name), as_grey=True)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(os.path.join(data_path,'imgs_train.npy'), imgs)
    np.save(os.path.join(data_path,'imgs_mask_train.npy'), imgs_mask)
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load(os.path.join(data_path,'train','imgs_train.npy'))
    imgs_mask_train = np.load(os.path.join(data_path,'train','imgs_mask_train.npy'))
    return imgs_train, imgs_mask_train


def create_test_data():
    train_data_path = os.path.join(data_path, 'test/image')
    images = os.listdir(train_data_path)
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        img_id = int(image_name.split('.')[0])
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)

        img = np.array([img])

        imgs[i] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(os.path.join(data_path,'test','imgs_test.npy'), imgs)
    np.save(os.path.join(data_path,'test','imgs_id_test.npy'), imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load(os.path.join(data_path,'test','imgs_test.npy'))
    imgs_id = np.load(os.path.join(data_path,'test','imgs_id_test.npy'))
    return imgs_test, imgs_id

if __name__ == '__main__':
    create_train_far_data()
    create_test_data()
