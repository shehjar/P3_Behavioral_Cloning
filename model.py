# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 13:11:41 2017

@author: admin
"""

from keras.models import Sequential, load_model
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Flatten
from keras.layers.advanced_activations import ELU
# Fix error with Keras and TensorFlow
#import tensorflow as tf
#tf.python.control_flow_ops = tf

def NVidia_Model(shape = (66,200,3)):
    '''Return the CNN architecture used by NVidia'''
    model = Sequential()
    model.add(BatchNormalization(input_shape = shape))
    ConvLayerDepth_5_5 = [24,36,48]
    ConvLayerDepth_3_3 = [64,64]
    DenselayerDepth = [1164,100,50,10]
    for depth in ConvLayerDepth_5_5:
        model.add(Convolution2D(depth,5,5,init='he_normal',subsample=(2,2)))
        #model.add(MaxPooling2D())
        model.add(ELU())
    for depth in ConvLayerDepth_3_3:
        model.add(Convolution2D(depth,3,3,init='he_normal'))
        model.add(ELU())
    model.add(Flatten())
    for depth in DenselayerDepth:
        model.add(Dense(depth,init='he_normal'))
        model.add(ELU())
    model.add(Dense(1,init='he_normal'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model