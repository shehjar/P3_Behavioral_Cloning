# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 13:11:41 2017

@author: admin
"""

from keras.models import Sequential, load_model
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.advanced_activations import ELU
from keras.layers import Cropping2D, Lambda
from keras.optimizers import Adam
from keras.regularizers import l2
# Fix error with Keras and TensorFlow
#import tensorflow as tf
#tf.python.control_flow_ops = tf

def NVidia_Model(shape = (160,320,3)):
    '''Return the CNN architecture used by NVidia'''
    #shape = (66,200,3)
    model = Sequential()
    #model.add(Cropping2D(cropping=((int(shape[0]/2),0),(0,0)), input_shape=shape))
    model.add(Cropping2D(cropping=((int(shape[0]/5),25),(0,0)), input_shape=shape))
    #model.add(BatchNormalization(input_shape = shape))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    ConvLayerDepth_5_5 = [24,36,48]
    ConvLayerDepth_3_3 = [64,64]
    DenselayerDepth = [1164,100,50,10]
    for depth in ConvLayerDepth_5_5:
        #print(model.output_shape)
        #print(depth)
        model.add(Convolution2D(depth,5,5,init='he_normal',subsample=(2,2)))
        model.add(ELU())
        #model.add(Dropout(0.5))
    for depth in ConvLayerDepth_3_3:
        model.add(Convolution2D(depth,3,3,init='he_normal'))    #, W_regularizer = l2(0.001)
        model.add(ELU())
        #model.add(Dropout(0.5))
    model.add(Flatten())
    #model.add(Dropout(0.5))
    for depth in DenselayerDepth:
        model.add(Dense(depth,init='he_normal')) #, W_regularizer = l2(0.0025)
        model.add(ELU())
        #if depth in DenselayerDepth[:2]:
        #model.add(Dropout(0.5))
    model.add(Dense(1,init='he_normal',activation='linear'))
    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss='mean_squared_error')
    return model