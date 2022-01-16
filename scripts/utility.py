import numpy as np
import nibabel as nib
import torch
import math
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
import matplotlib.pyplot as plt

def load_data_nii(fname):
    import nibabel as nib
    img = nib.load(fname)
    affine_mat=img.affine
    hdr=img.header
    data = img.get_fdata()
    data_norm = data
    return data_norm 
    return 1

def calculate_l1_distance(img,img2):
    dist = np.sum(abs(img[:] - img2[:]));
    return dist

def calculate_l2_distance(img,img2):
    dist = np.sqrt(np.sum((img[:] - img2[:])** 2));
    return dist

def calculate_RMSE(img,img2):
    m=img.shape[0]
    n= img.shape[1]
    rmse = np.sqrt(np.sum((img[:] - img2[:])** 2)/(m*n));
    return rmse

def calculate_PSNR(img,img2):
    psnr = 10* math.log10( (np.sum(img[:]** 2)) / (np.sum((img[:] - img2[:])** 2)) )
    return psnr


def fourier_transform(image,shift=False):
    FT = np.fft.fft2(image)
    
    if shift:
        f_shift = np.fft.fftshift(FT)
        return f_shift
    else:
        return FT
    
def inverse_fourier_transform(data,shift=False):
    if shift:
        image = np.fft.ifft2(np.fft.ifftshift(data))
    else:
        image = np.fft.ifft2(data)
    return image

def save_img(img,name,fol_dir):
    figure(figsize=(8, 6), dpi=80)
    plt.imshow(img, cmap = 'gray')
    plt.axis('off')
    plt.savefig(fol_dir+name+'.png', bbox_inches = 'tight',facecolor='white',pad_inches = 0) 
    plt.show()

def save_img_using_pil_lib(img,name,fol_dir):
    data= img
    data = data.astype('float')
    data = (data/data.max())*255
    data = data.astype(np.uint8)
    data = Image.fromarray(data)
    data.save(fol_dir+name+'.png')
    
def crop_center(arr,factor):
    y,x = arr.shape
    center_y = y//2 #defining the center of image in x and y direction
    center_x = x//2
    startx = center_x-(x//(factor*2))  
    starty = center_y-(y//(factor*2))
    return arr[starty:starty+(y//factor),startx:startx+(x//factor)]  

def pad_zeros_around(arr,factor,original_arr):
    y,x = original_arr.shape
    rows_pad =(y-(y//factor))//2
    cols_pad =(x-(x//factor))//2
    return np.pad(arr, [(rows_pad, rows_pad), (cols_pad, cols_pad)], mode='constant',constant_values=0)

def crop_pad_image_kspace(data,pad=False,factor=2):  #function for cropping and/or padding the image in kspace
    F = np.fft.fft2(data)
    fshift = np.fft.fftshift(F)
    data = crop_center(arr=fshift,factor=factor)
    
    if pad:
        data= pad_zeros_around(data,factor,fshift)
    img_reco_cropped = np.fft.ifft2(np.fft.ifftshift(data))
    return np.abs(img_reco_cropped )

def normalize_image(image):
    max_img = image.max()
    min_img = image.min()
    denom = max_img-min_img
    norm_image = (image-min_img)/denom
    return norm_image

def get_gauss_filter(image,R=30,high=False):
    X = [i for i in range(image.shape[1])]
    Y = [i for i in range(image.shape[0])]
    
    Cy, Cx = image.shape
    val = 0.5

    X,Y = np.meshgrid(X, Y)
    low_filter = np.exp(-((X-(Cx*val))**2+(Y-(Cy*val))**2)/(2*R)**2)
    if high:
        return (1-low_filter)
    else:
        return low_filter