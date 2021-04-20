import numpy as np
import os, glob
import cv2
import imageio
from math import log10
import torch
from skimage.measure.simple_metrics import compare_psnr

def psnr(x, target):
    sqrdErr = np.mean((x - target) ** 2)
    return 10 * log10(1/sqrdErr)

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def batch_mse(pred, disp):
    pred = pred.data.cpu().numpy().astype(np.float32)
    disp = disp.data.cpu().numpy().astype(np.float32)

    return np.mean(np.square(pred-disp))
