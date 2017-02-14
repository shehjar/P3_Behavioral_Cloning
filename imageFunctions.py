# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 11:17:40 2017

@author: admin
"""
import cv2
import random
import numpy as np

def resizing(image,shape=(200,66)):
    height,width = image.shape[:2]
    image = image[30:height-25,:,:]
    resize_img = cv2.resize(image,dsize=shape,interpolation= cv2.INTER_AREA)
    return resize_img

def translation_augmentation(image,angle,range_translation = 100):
    height,width = image.shape[:2]
    angle_per_pixel_shift = 0.004
    x_trans_value = np.random.uniform(-range_translation/2,range_translation/2)
    angle_aug = angle + x_trans_value*angle_per_pixel_shift
    y_trans_value = np.random.uniform(-20,20)
    M = np.float32([[1,0,x_trans_value],[0,1,y_trans_value]])
    aug_image = cv2.warpAffine(image,M,(width,height))
    return aug_image, angle_aug

def brightness_augmentation(image, flag = 'uniform'):
    height,width = image.shape[:2]
    HSV_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    #h,s,v = cv2.split(HSV_image)
    if flag == 'shadow':
        multiplier = 0.5
    else:
        multiplier = np.random.uniform()+0.25
    HSV_image[:,:,2] = cv2.multiply(HSV_image[:,:,2],np.array([multiplier]))
    return cv2.cvtColor(HSV_image,cv2.COLOR_HSV2BGR)

def shadow_augmentation(image):
    shadow_image = brightness_augmentation(image,'shadow')
    height,width = image.shape[:2]
    pt1 = [random.randint(0,width),0]
    pt2 = [random.randint(0,width),height]
    polygon = np.array([[0,0],pt1,pt2,[0,height]])
    # Create mask template
    mask = image.copy()
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    mask.fill(0)
    if np.random.uniform() > 0.5:
        src1 = image
        src2 = shadow_image
    else:
        src1 = shadow_image
        src2 = image
    # fill polygon in mask
    cv2.fillConvexPoly(mask,polygon,255)
    # create region of interest
    mask_inv = cv2.bitwise_not(mask)
    img_part1 = cv2.bitwise_and(src1,src1,mask=mask_inv)
    img_part2 = cv2.bitwise_and(src2,src2,mask=mask)
    final_img = cv2.add(img_part1,img_part2)
    return final_img