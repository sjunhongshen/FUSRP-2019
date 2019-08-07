import PIL
import scipy, shutil, os, nibabel
import sys, getopt
from matplotlib.pyplot import imsave
from PIL import Image
import cv2
import numpy as np
from sklearn.preprocessing import normalize

# input: batch_size * channel * height * width
# output: batch_size * height * width * channel
# clip: 100, power: 3

def readNIfTI(filename):
    image_array = nibabel.load(filename).get_data()
    if len(image_array.shape) == 3:
        output = np.reshape(np.transpose(image_array, (2, 0, 1)), (image_array.shape[2], image_array.shape[0], image_array.shape[1], 1))
    elif len(image_array.shape) == 4:
        output = np.reshape(np.transpose(image_array, (0, 3, 1, 2)), (image_array.shape[0], image_array.shape[3], image_array.shape[1], image_array.shape[2], 1))
    else:
        print("image array shape:")
        print(image_array.shape)
    return np.transpose(image_array, (2, 0, 1)), nibabel.load(filename).affine

def writeNIfTI(data, path, affine):
    img = nibabel.Nifti1Image(data, affine)
    nibabel.save(img, path)
    return

def rgb2gray(rgb):
    assert (len(rgb.shape) == 4)
    assert (rgb.shape[3] == 3)
    bn_imgs = rgb[:, :, :, 0] * 0.299 + rgb[:, :, :, 1] * 0.587 + rgb[:, :, :, 2] * 0.114
    bn_imgs = np.reshape(bn_imgs, (rgb.shape[0], rgb.shape[1], rgb.shape[2], 1))
    return bn_imgs

#group a set of images row per columns
def group_images(data, per_row):
    assert (data.shape[0] % per_row == 0)
    assert (data.shape[3] == 1 or data.shape[3] == 3)
    all_stripe = []
    for i in range(int(data.shape[0] / per_row)):
        stripe = data[i * per_row]
        for k in range(i * per_row + 1, i * per_row + per_row):
            stripe = np.concatenate((stripe, data[k]), axis = 1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1, len(all_stripe)):
        totimg = np.concatenate((totimg, all_stripe[i]), axis = 0)
    return totimg

#visualize image (as PIL image, NOT as matplotlib!)
def visualize(data, filename):
    assert (len(data.shape) == 3) #height*width*channels
    img = None
    if data.shape[2] == 1:  #in case it is black and white
        data = np.reshape(data, (data.shape[0], data.shape[1]))
    if np.max(data) > 1:
        img = Image.fromarray(data.astype(np.uint8))   #the image is already 0-255
    else:
        img = Image.fromarray((data * 255).astype(np.uint8))  #the image is between 0-1
    img.save(filename + '.png')
    return img

#prepare the mask in the right shape for the Unet
def masks_Unet(masks):
    assert (len(masks.shape) == 4)  #4D arrays
    assert (masks.shape[3] == 1)  #check the channel is 1
    im_h = masks.shape[1]
    im_w = masks.shape[2]
    masks = np.reshape(masks, (masks.shape[0], im_h * im_w))
    new_masks = np.empty((masks.shape[0], im_h * im_w, 2))
    for i in range(masks.shape[0]):
        for j in range(im_h * im_w):
            if masks[i,j] == 0:
                new_masks[i, j, 0] = 1
                new_masks[i, j, 1] = 0
            else:
                new_masks[i, j, 0] = 0
                new_masks[i, j, 1] = 1
    return new_masks


def pred_to_imgs(pred, patch_height, patch_width, mode="original"):
    assert (len(pred.shape) == 3)  #3D array: (Npatches,height*width,2)
    assert (pred.shape[2] == 2)  #check the classes are 2
    pred_images = np.empty((pred.shape[0], pred.shape[1]))  #(Npatches,height*width)
    if mode == "original":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                pred_images[i, pix] = pred[i, pix, 1]
    elif mode == "threshold":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                if pred[i, pix, 1] >= 0.5:
                    pred_images[i, pix] = 1
                else:
                    pred_images[i, pix] = 0
    else:
        print("mode " + str(mode) + " not recognized, it can be 'original' or 'threshold'")
        exit()
    pred_images = np.reshape(pred_images, (pred_images.shape[0], 1, patch_height, patch_width))
    return pred_images

#============================================================
#========= BATCH PRE PROCESSING FUNCTIONS ========================#
#============================================================

#==== histogram equalization
def histo_equalized(imgs):
    assert (len(imgs.shape) == 4)  #4D arrays
    assert (imgs.shape[1] == 1)  #check the channel is 1
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i, 0] = cv2.equalizeHist(np.array(imgs[i, 0], dtype = np.uint8))
    return imgs_equalized

def clip_intensity(imgs, upper_limit):
    #imgs = np.clip(imgs, 0, upper_limit)
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs - imgs_mean) / imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i]) - np.min(imgs_normalized[i])))
    return imgs_normalized

def raise_power(imgs, power):
    imgs_power = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        #imgs_power[i, 0] = imgs[i, 0] ** power
        imgs_power[i] = imgs[i] ** power
    return imgs_power

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
#adaptive histogram equalization is used. In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied
def clahe_equalized(imgs):
    assert (len(imgs.shape) == 4)  #4D arrays
    assert (imgs.shape[1] == 1)  #check the channel is 1
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = clahe.apply(np.array(imgs[i, 0], dtype = np.uint8))
    return imgs_equalized

# ===== normalize over the dataset
def dataset_normalized(imgs):
    assert (len(imgs.shape) == 4)  #4D arrays
    assert (imgs.shape[1] == 1)  #check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs - imgs_mean) / imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i]) - np.min(imgs_normalized[i]))) * 255
    return imgs_normalized

def adjust_gamma(imgs, gamma = 1.0):
    assert (len(imgs.shape) == 4)  #4D arrays
    assert (imgs.shape[1] == 1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i, 0] = cv2.LUT(np.array(imgs[i, 0], dtype = np.uint8), table)
    return new_imgs

def preprocess(imgs):
    #imgs = np.transpose(imgs, (0, 3, 1, 2))
    imgs_normalized = clip_intensity(imgs, 200)
    imgs_power = raise_power(imgs_normalized, 3)
    #return np.transpose(imgs_power, (0, 2, 3, 1))
    return np.transpose(imgs_power, (1, 2, 0))

import numpy as np

from utils import divide_nonzero
from hessian import absolute_hessian_eigenvalues


def frangi(nd_array, scale_range=(1, 10), scale_step=2, alpha=0.5, beta=0.5, frangi_c=100, black_vessels=True):

    if not nd_array.ndim == 3:
        raise(ValueError("Only 3 dimensions is currently supported"))

    # from https://github.com/scikit-image/scikit-image/blob/master/skimage/filters/_frangi.py#L74
    sigmas = np.arange(scale_range[0], scale_range[1], scale_step)
    if np.any(np.asarray(sigmas) < 0.0):
        raise ValueError("Sigma values less than zero are not valid")

    filtered_array = np.zeros(sigmas.shape + nd_array.shape)

    for i, sigma in enumerate(sigmas):
        eigenvalues = absolute_hessian_eigenvalues(nd_array, sigma=sigma, scale=True)
        filtered_array[i] = compute_vesselness(*eigenvalues, alpha=alpha, beta=beta, c=frangi_c,
                                               black_white=black_vessels)

    return np.max(filtered_array, axis=0)


def compute_measures(eigen1, eigen2, eigen3):
    """
    RA - plate-like structures
    RB - blob-like structures
    S - background
    """
    Ra = divide_nonzero(np.abs(eigen2), np.abs(eigen3))
    Rb = divide_nonzero(np.abs(eigen1), np.sqrt(np.abs(np.multiply(eigen2, eigen3))))
    S = np.sqrt(np.square(eigen1) + np.square(eigen2) + np.square(eigen3))
    return Ra, Rb, S


def compute_plate_like_factor(Ra, alpha):
    return 1 - np.exp(np.negative(np.square(Ra)) / (2 * np.square(alpha)))


def compute_blob_like_factor(Rb, beta):
    return np.exp(np.negative(np.square(Rb) / (2 * np.square(beta))))


def compute_background_factor(S, c):
    return 1 - np.exp(np.negative(np.square(S)) / (2 * np.square(c)))


def compute_vesselness(eigen1, eigen2, eigen3, alpha, beta, c, black_white):
    Ra, Rb, S = compute_measures(eigen1, eigen2, eigen3)
    plate = compute_plate_like_factor(Ra, alpha)
    blob = compute_blob_like_factor(Rb, beta)
    background = compute_background_factor(S, c)
    return filter_out_background(plate * blob * background, black_white, eigen2, eigen3)


def filter_out_background(voxel_data, black_white, eigen2, eigen3):
    """
    Set black_white to true if vessels are darker than the background and to false if
    vessels are brighter than the background.
    """
    if black_white:
        voxel_data[eigen2 < 0] = 0
        voxel_data[eigen3 < 0] = 0
    else:
        voxel_data[eigen2 > 0] = 0
        voxel_data[eigen3 > 0] = 0
    voxel_data[np.isnan(voxel_data)] = 0
    return voxel_data

def preprocess_img(filename, savepath):
    # filename = '/Users/kimihirochin/Desktop/mesh/IXI002-Guys-0828-ANGIOSENS_-s256_-0701-00007-000001-01.nii'
    # savepath = '/Users/kimihirochin/Desktop/mesh/sample1_changed.nii.gz'
    imgs, aff = readNIfTI(filename)
    imgs_preprocessed = preprocess(imgs)
    writeNIfTI(imgs_preprocessed, savepath, aff)


if __name__ == "__main__":
    filename = '/Users/kimihirochin/Desktop/mesh/IXI002-Guys-0828-ANGIOSENS_-s256_-0701-00007-000001-01.nii'
    savepath = '/Users/kimihirochin/Desktop/mesh/sample1_changed.nii.gz'
    savepath2 = '/Users/kimihirochin/Desktop/mesh/prob.nii.gz'
    imgs, aff = readNIfTI(filename)
    print(aff)
    imgs_preprocessed = preprocess(imgs)
    writeNIfTI(imgs_preprocessed, savepath, aff)
    exit()

    from nipype.interfaces import brainsuite
    from nipype.testing import example_data
    bse = brainsuite.Bse()
    print(bse)
    bse.inputs.inputMRIFile = filename
    results = bse.run() 
    print(results)
    imgs = nibabel.load(savepath2).get_data()
    print(imgs.shape)
    #imgs = np.squeeze(imgs, axis=(3,))
    #ret = frangi(imgs)
    ret = imgs
    print(ret)        
    print(np.max(ret))
    print(np.count_nonzero(ret > 0.1))
    ret = ret / np.max(ret)
    var = np.var(ret)
    print(ret.shape)        
    print(np.max(ret))
    print(np.count_nonzero(ret != 0))
    print(var)
    exit()




    imgs_preprocessed = preprocess(imgs)
    writeNIfTI(imgs_preprocessed, savepath)
    for i in range(imgs_preprocessed.shape[0]):
        visualize(imgs_preprocessed[i], filename + "_{:0>3}".format(str(i)))
    #print(img_normalized.shape)
    #print(np.sum((img_normalized), axis=0))
    exit()
