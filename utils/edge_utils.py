"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import numpy as np
from PIL import Image
import cv2
from scipy.ndimage.morphology import distance_transform_edt

def mask_to_onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _mask = [mask == i for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)

def onehot_to_mask(mask, background=255):
    """
    Converts a mask (K,H,W) to (H,W)
    """
    _mask = np.argmax(mask, axis=0)
    _mask[mask.sum(0)==0] = background
    return _mask

def onehot_to_multiclass_edges(mask, radius_max, num_classes, radius_min=-1):
    """
    Converts a segmentation mask (K,H,W) to an edgemap (K,H,W)

    """
    if radius_max < 0:
        return mask
    
    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    
    edgemap = np.zeros(mask.shape)
    ind = np.where(mask.sum(axis=(1, 2)) > 0)[0]
    for i in ind:
        dist = distance_transform_edt(mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        if radius_min>=0:
            dist[dist <= radius_min] = 0
        dist[dist > radius_max] = 0
        dist = (dist > 0).astype(np.uint8)
        edgemap[i] = dist
        
    return edgemap

def onehot_to_binary_edges(mask, radius_max, num_classes, radius_min=-1):
    """
    Converts a segmentation mask (K,H,W) to a binary edgemap (1,H,W)
    """
    if radius_max < 0:
        return mask
    
    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    
    edgemap = np.zeros(mask.shape[1:])
    ind = np.where(mask.sum(axis=(1, 2)) > 0)[0]
    for i in ind:
        dist = distance_transform_edt(mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        if radius_min>=0:
            dist[dist <= radius_min] = 0
        dist[dist > radius_max] = 0
        edgemap += dist
    edgemap = np.expand_dims(edgemap, axis=0)    
    edgemap = (edgemap > 0).astype(np.uint8)
    return edgemap

def onehot_to_binary_edges_old(mask, radius_max, num_classes, radius_min=-1):
    """
    Converts a segmentation mask (K,H,W) to a binary edgemap (1,H,W)
    """
    if radius_max < 0:
        return mask

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    edgemap = np.zeros(mask.shape[1:])

    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist <= radius_min] = 0
        dist[dist > radius_max] = 0
        edgemap += dist
    edgemap = np.expand_dims(edgemap, axis=0)
    edgemap = (edgemap > 0).astype(np.uint8)
    return edgemap

def binary_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode

def mask_to_boundary(mask, num_classes, dilation_ratio=0.02, background=255):
    onehot = mask_to_onehot(mask, num_classes).astype(np.uint8)
    
    onehot_boundary = np.zeros(onehot.shape, dtype=np.uint8)
    ind = np.where(onehot.sum(axis=(1, 2)) > 0)[0]
    for i in ind:
        onehot_boundary[i] = binary_to_boundary(onehot[i], dilation_ratio=dilation_ratio)
    
    mask_boundary = onehot_to_mask(onehot_boundary.astype(int),background=background)
    return mask_boundary
    