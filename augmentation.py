# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 15:42:03 2017
@author: admin

"""

import cv2, os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Flatten
from keras.layers.advanced_activations import ELU
from model import NVidia_Model

CurFolder = os.getcwd()
ParFolder = os.path.abspath(os.path.join(CurFolder, os.pardir))
DataFolder = os.path.join(ParFolder,'data')
TrainingFolder = os.path.join(DataFolder,'Training3_beta')
csvFile = os.path.join(TrainingFolder,'driving_log.csv')

def saving_file(CurFolder):
    saving_dir = os.path.join(CurFolder,'Model_Save')
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)
    return os.path.join(saving_dir,'model.h5')

def GetData(csvFile):
    Data = pd.read_csv(csvFile, names=['Center Image','Left Image', 'Right Image', 'Steering Angle', 'Throttle', 'Break', 'Speed'])
    imageFiles = list(Data['Center Image'])
    images = []
    for imgpath in imageFiles:
        images.append(cv2.imread(imgpath))
    images = np.array(images)
    angles = Data['Steering Angle'].as_matrix()
    return images,angles
    
def preprocessing(image):
    resize_img = cv2.resize(image,dsize=(200,66),interpolation= cv2.INTER_AREA)
    return resize_img

def processImageArray(Images):
    new_images = []
    for i in range(len(Images)):
        new_images.append(preprocessing(Images[i,:,:,:]))
    return np.array(new_images)

def generator(batch_size, X, y):
    '''Generate batches forever'''
    while True:
        batch_X, batch_y = [],[]
        ind_set = []
        for i in range(batch_size):
            while True:
                rand_index = random.randint(0,len(X)-1)
                if rand_index not in ind_set:
                    ind_set.append(rand_index)
                    break
            image = X[rand_index,:,:,:]
            batch_X.append(preprocessing(image))
            batch_y.append(y[rand_index])
        yield np.array(batch_X), np.array(batch_y)

model = NVidia_Model()
X,y = GetData(csvFile)
X,y = shuffle(X,y)
train_X, valid_X, train_y, valid_y = train_test_split(X,y,train_size=0.8)

# Preprocessing the validation images
valid_X = processImageArray(valid_X)

model.fit_generator(generator(64,train_X,train_y),samples_per_epoch=64,nb_epoch=5,validation_data=(valid_X,valid_y))
model.save(saving_file(CurFolder))
# Convert into YUV
#with open(csvFile, 'rb') as f:
#Data = pd.read_csv(csvFile, names=['Center Image','Left Image', 'Right Image', 'Steering Angle', 'Throttle', 'Break', 'Speed'])
#Photo_index = 0

# Crop Images and transform to RGB
#image = cv2.imread(Data['Center Image'][Photo_index])
#b,g,r = cv2.split(image)
#RGB_image = cv2.merge([r,g,b])
#height,width,depth = RGB_image.shape
#crop_height = int(np.ceil(height/5))
#crop_image = RGB_image[crop_height:height-crop_height,:,:]

# Plot image
#plt.imshow(crop_image)
