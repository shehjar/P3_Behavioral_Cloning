# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 10:48:08 2017

@author: admin
"""

import cv2, os, json
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from model import NVidia_Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from imageFunctions import resizing, translation_augmentation, brightness_augmentation, shadow_augmentation
from model import GetData, preprocessing_training, preprocessing_testing, FLAGS, saving_file
CurFolder = os.getcwd()
ParFolder = os.path.abspath(os.path.join(CurFolder, os.pardir))
DataFolder = os.path.join(ParFolder,'data')
TrainingFolder = os.path.join(DataFolder,'data')
csvFile = os.path.join(TrainingFolder,'driving_log.csv')

ModelPath = os.path.join(CurFolder,'Model_Save')
ModelFile = os.path.join(ModelPath,'model.h5')

def generator_train(X, y, batch_size=256):
    '''Generate batches forever
    The images are returned in a multiple of 8 after the preprocessing'''
    while True:
        batch_X, batch_y = [],[]
        X,y = shuffle(X,y)
        prob = np.absolute(y+0.01)/np.sum(np.absolute(y+0.01))
        rand_index = np.random.choice(len(X),size=batch_size,replace=False,p=prob)
        images = [X[i] for i in rand_index]
        angles = y[rand_index]
        for ind in range(len(images)):
            pre_image,pre_angle = preprocessing_training(images[ind],angles[ind])
            batch_X.append(pre_image)
            batch_y.append(pre_angle)
        yield np.array(batch_X), np.array(batch_y)

def main(_):
    model = load_model(ModelFile)
    X,y = GetData(csvFile)
    X,y = shuffle(X,y)
    train_X, valid_X, train_y, valid_y = train_test_split(X,y,train_size=0.9)
    #train_X, train_y = X,y
    batch_size = FLAGS.batch_size
    valid_X = preprocessing_testing(valid_X)
    dir_save = saving_file(CurFolder)
    epoch_save_file = os.path.join(dir_save,'model.{epoch:04d}-{val_loss:.2f}.h5')
    weight_save_callback = ModelCheckpoint(epoch_save_file, monitor='val_loss', verbose=0, 
                                           save_best_only=False, mode='auto',period=FLAGS.save_interval)
    stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='min')
    model.summary()
    model.fit_generator(generator_train(train_X,train_y,batch_size),
                        samples_per_epoch=FLAGS.sample_size,nb_epoch=FLAGS.epochs,
                        validation_data=(valid_X,valid_y), callbacks=[weight_save_callback,stopping])
    #weight_save_callback = ModelCheckpoint(epoch_save_file, monitor='val_loss', verbose=0, save_best_only=False, mode='auto')
    #model.fit(X_train,y_train,batch_size=batch_size,nb_epoch=nb_epoch,callbacks=[weight_save_callback])
    #model.fit_generator(generator(train_X,train_y,prob_dist,batch_size),samples_per_epoch=batch_size,nb_epoch=1000)
    model.save(os.path.join(dir_save,'model.h5'))
    with open(os.path.join(dir_save,'model.json'),'w') as json_file:
        json.dump(model.to_json(), json_file)
        
if __name__ == '__main__':
    tf.app.run()