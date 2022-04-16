import os
import numpy as np
from skimage.io import imread, imsave
import sys
import math

def Read_img(path: str):
    img = imread(path)
    if len(img.shape) == 3:
        img = img[:,:,0]
    return img.astype(float)
    
def Mse(img1: np.array, img2: np.array):
    return np.mean((img1 - img2) ** 2)
    
def Psnr(img1: np.array, img2: np.array):
    sqrt_mse = np.sqrt(Mse(img1, img2))
    if sqrt_mse == 0:
        return float("inf")
    return 20 * np.log10(255 / sqrt_mse)
    
def Ssim(img1: np.array, img2: np.array, k1 = 1e-2, k2 = 3e-2, L = 255):
    mean1 = np.mean(img1)
    mean2 = np.mean(img2)
    var1 = np.std(img1) ** 2
    var2 = np.std(img2) ** 2
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    cov = np.mean(img1 * img2) - mean1 * mean2
    
    top = (2 * mean1 * mean2 + c1) * (2 * cov + c2)
    bottom = (mean1 ** 2 + mean2 ** 2 + c1) * (var1 + var2 + c2) 
    return top / bottom
    
def Pad(img: np.array, size: int):
    pad_size = ((size, size), (size, size))
    return np.pad(img, pad_size, 'edge')

def Median(img: np.array, rad: int):
    output = np.zeros_like(img, dtype=np.uint8)
    padded_img = Pad(img, rad)
    height, width = img.shape
    for y in range(height):
        for x in range(width):
            px = x + rad
            py = y + rad
            crop = padded_img[py-rad:py+rad+1, px-rad:px+rad+1]
            output[y,x] = int(np.median(crop))
    return output
    
def GetKernel(sigma: float):
    size = math.ceil(3 * sigma) * 2 + 1
    rad = size // 2
    x = np.arange(-rad, rad + 1, dtype=float)
    y = np.arange(-rad, rad + 1, dtype=float)[:,np.newaxis]
    exp = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    gaussian = 1 / (2 * np.pi * sigma * sigma) * exp
    return gaussian / np.sum(gaussian)
    
def Gauss(img: np.array, sigma: float):
    output = np.zeros_like(img, dtype=np.uint8)
    kernel = GetKernel(sigma)
    rad = kernel.shape[0] // 2
    padded_img = Pad(img, rad)
    height, width = img.shape
    for y in range(height):
        for x in range(width):
            px = x + rad
            py = y + rad
            crop = padded_img[py-rad:py+rad+1, px-rad:px+rad+1]
            output[y,x] = int(np.sum(crop * kernel))
    return output
    
def ExpFunc(mat: np.array, sigma: float):
    return np.exp(-mat * mat / (2 * sigma * sigma))

def Bilateral(img: np.array, sigma_d: float, sigma_r: float):
    output = np.zeros_like(img, dtype=np.uint8)
    kernel = GetKernel(sigma_d)
    rad = kernel.shape[0] // 2
    padded_img = Pad(img, rad)
    height, width = img.shape
    for y in range(height):
        for x in range(width):
            px = x + rad
            py = y + rad
            crop = padded_img[py-rad:py+rad+1, px-rad:px+rad+1]
            intensity = abs(crop - crop[rad, rad])
            intensity_gaussian = ExpFunc(intensity, sigma_r)
            weights = intensity_gaussian * (kernel * 2 * np.pi * sigma_d * sigma_d)
            val = np.sum(crop * weights) / np.sum(weights)
            output[y, x] = int(val)
    return output

if __name__ == '__main__':
    op = sys.argv[1]
    input_file1 = sys.argv[-2]
    input_file2 = sys.argv[-1] 
    if op == 'mse': 
        img1 = Read_img(input_file1)
        img2 = Read_img(input_file2)
        print(Mse(img1, img2))
    elif op == 'psnr': 
        img1 = Read_img(input_file1)
        img2 = Read_img(input_file2)
        print(Psnr(img1, img2))
    elif op == 'ssim':
        img1 = Read_img(input_file1)
        img2 = Read_img(input_file2)
        print(Ssim(img1, img2))
    elif op == 'median':
        img = Read_img(input_file1)
        rad = int(sys.argv[-3])
        imsave(input_file2, Median(img, rad))
    elif op == 'gauss':
        img = Read_img(input_file1)
        sigma = float(sys.argv[-3])
        imsave(input_file2, Gauss(img, sigma))
    else:
        img = Read_img(input_file1)
        sigma_d = float(sys.argv[-4])
        sigma_r = float(sys.argv[-3])
        imsave(input_file2, Bilateral(img, sigma_d, sigma_r))
