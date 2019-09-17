"""
Author: Junhong Shen (jhshen@g.ucla.edu)

Description: Clinical image preprocessing.
"""

# input: batch_size * channel * height * width
# output: batch_size * height * width * channel
# clip: 100, power: 3

import PIL
import scipy, shutil, os, nibabel
import sys, getopt
from matplotlib.pyplot import imsave
from PIL import Image
import cv2
import numpy as np
from sklearn.preprocessing import normalize


def readNIfTI(filename):
    """ 
    Converts NIfTI file to 3D array.
    """

    image_array = nibabel.load(filename).get_data()
    if len(image_array.shape) == 3:
        output = np.reshape(np.transpose(image_array, (2, 0, 1)), (image_array.shape[2], image_array.shape[0], image_array.shape[1], 1))
    elif len(image_array.shape) == 4:
        output = np.reshape(np.transpose(image_array, (0, 3, 1, 2)), (image_array.shape[0], image_array.shape[3], image_array.shape[1], image_array.shape[2], 1))
    return np.transpose(image_array, (2, 0, 1)), nibabel.load(filename).affine


def writeNIfTI(data, path, affine):
    """ 
    Converts 3D array to NIfTI file and saves as image.
    """

    img = nibabel.Nifti1Image(data, affine)
    nibabel.save(img, path)
    return


def clip_intensity(imgs, upper_limit):
    """ 
    Clips image intensity above a threshold and normalizes the image.
    """

    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs - imgs_mean) / imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i]) - np.min(imgs_normalized[i])))
    return imgs_normalized


def raise_power(imgs, power):
    """ 
    Increases image contrast by raising the intensity to a power of 2 or 3.
    """

    imgs_power = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_power[i] = imgs[i] ** power
    return imgs_power


def preprocess(imgs):
    """ 
    First clips the intensity of the image, then increases the contrast.
    """

    imgs_normalized = clip_intensity(imgs, 200)
    imgs_power = raise_power(imgs_normalized, 3)
    return np.transpose(imgs_power, (1, 2, 0))


if __name__ == "__main__":
    filename = 'IXI002-Guys-0828-ANGIOSENS_-s256_-0701-00007-000001-01.nii'
    savepath = 'test_1.nii.gz'
    imgs, aff = readNIfTI(filename)
    imgs_preprocessed = preprocess(imgs)
    writeNIfTI(imgs_preprocessed, savepath, aff)
    