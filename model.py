# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 22:07:39 2017

@author: admin
"""

import cv2, os, pickle,time
import json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras_model import NVidia_Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from imageFunctions import translation_augmentation, brightness_augmentation, shadow_augmentation

# Flags for automated procedure
flags = tf.app.flags
FLAGS = flags.FLAGS

# Command line flags
flags.DEFINE_integer('epochs', 15, "The number of epochs.")
flags.DEFINE_integer('batch_size', 64, "The batch size for training data.")
flags.DEFINE_integer('sample_size', 19200, "The total number of training samples")
flags.DEFINE_float('corr_angle', 0.35, "The correction angle for left and right camera images")
flags.DEFINE_integer('save_interval', 2, "The interval between two saved files.")

# Define paths for getting data
CurFolder = os.getcwd()
ParFolder = os.path.abspath(os.path.join(CurFolder, os.pardir))
DataFolder = os.path.join(ParFolder,'data')
TrainingFolder = os.path.join(DataFolder,'data')
csvFile = os.path.join(TrainingFolder,'driving_log.csv')

def strip(text):
    ''' To remove spaces between text strings within csv file'''
    try:
        return text.strip()
    except AttributeError:
        return text

def saving_file(CurFolder):
    '''Create a svaing folder for saving the neural network model
    Input -> Current Folder Path
    Output -> Saving Folder Path'''
    saving_dir = os.path.join(CurFolder,'Model_Save')
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)
    return saving_dir

def GetData(csvFile):
    '''Extract Data from the CSV file.
    Input -> CSV file path
    Output -> (image (Left, Right, Center) list, steering angles numpy array)'''
    Par_dir = os.path.abspath(os.path.join(csvFile,os.pardir))
    Data = pd.read_csv(csvFile,converters={'center' : strip, 'left': strip, 'right':strip})
    imageFilesCenter = Data['center'].tolist()
    imageFilesLeft = Data['left'].tolist()
    imageFilesRight = Data['right'].tolist()
    center_angles = Data['steering'].tolist()
    images,angles = [],[]
    for Center_imgpath,Left_imgpath, Right_imgpath, ang in zip(imageFilesCenter,imageFilesLeft,imageFilesRight, center_angles):
        Full_path_center = os.path.join(Par_dir,Center_imgpath)
        Full_path_left = os.path.join(Par_dir,Left_imgpath)
        Full_path_right = os.path.join(Par_dir,Right_imgpath)
        images.append([Full_path_center,Full_path_left,Full_path_right])
        angles.append(ang)
    angles = np.array(angles)
    return images,angles

def preprocessing_training(imagePath, angle):
    '''Create batches of modified images through image augmentation procedures
    Input -> (Image paths for left, right and center images, steering angle)
    Output -> (List of augmented flipped images and their corresponding angles)'''
    img = imagePath[0]
    L_img = imagePath[1]
    # addiing correction angle for the camera view
    L_angle = angle + FLAGS.corr_angle
    R_img = imagePath[2]
    R_angle = angle - FLAGS.corr_angle
    # Reading the images
    image = cv2.imread(img)
    L_image = cv2.imread(L_img)
    R_image = cv2.imread(R_img)
    # Flag for brightness or shadow augmentation (only one of the two shall be carried out)
    br_flag = np.random.randint(2)
    # Applying translation augmentation
    image,angle = translation_augmentation(image,angle)
    L_image, L_angle = translation_augmentation(L_image, L_angle)
    R_image, R_angle = translation_augmentation(R_image, R_angle)
    # Limiting the value of normalized steering angle between -1 and 1
    angle = max(min(angle,1),-1)
    L_angle = max(min(L_angle,1),-1)
    R_angle = max(min(R_angle,1),-1)
    if br_flag:
        image = brightness_augmentation(image)
        L_image = brightness_augmentation(L_image)
        R_image = brightness_augmentation(R_image)
    else:
        image = shadow_augmentation(image)
        L_image = shadow_augmentation(L_image)
        R_image = shadow_augmentation(R_image)
    # Converting the images into HLS color space
    image = cv2.cvtColor(image,cv2.COLOR_BGR2HLS)
    L_image = cv2.cvtColor(L_image, cv2.COLOR_BGR2HLS)
    R_image = cv2.cvtColor(R_image, cv2.COLOR_BGR2HLS)
    # Flipping the respective images
    flip_image = cv2.flip(image,1)
    flip_angle = -1*angle
    flip_L_image = cv2.flip(L_image,1)
    flip_L_angle = -1*L_angle
    flip_R_image = cv2.flip(R_image,1)
    flip_R_angle = -1*R_angle
    return [image, flip_image, L_image, flip_L_image, R_image, flip_R_image],[angle,
           flip_angle,L_angle,flip_L_angle,R_angle,flip_R_angle]

def preprocessing_testing(imagePaths):
    '''Reading image data for testing dataset'''
    n_images = len(imagePaths)
    images=[]
    for i in range(n_images):
        image = cv2.imread(imagePaths[i][0])
        image = cv2.cvtColor(image,cv2.COLOR_BGR2HLS)
        images.append(image)
    return np.array(images)

def generator_train(X, y, batch_size=256):
    '''Generate batches forever
    The images are returned in a multiple of 6 after the preprocessing'''
    while True:
        batch_X, batch_y = [],[]
        X,y = shuffle(X,y)
        rand_index = np.random.choice(len(X),size=batch_size,replace=False)
        for i in range(batch_size):
            pre_images,pre_angles = preprocessing_training(X[int(rand_index[i])],y[rand_index[i]])
            batch_X.extend(pre_images)
            batch_y.extend(pre_angles)
        yield np.array(batch_X), np.array(batch_y)

def exponential_prob (b,x):
    '''Defines an exponential function'''
    return np.exp(b*x)

def equalizing_data(X,y,percent = 0.1):
    '''Including the almost 0 degree steering angle data within the main file'''
    X_out = [X[i] for i in range(len(X)) if abs(y[i]) >= 0.01]
    y_out = y[np.absolute(y) >= 0.01]
    low_turn = y[np.absolute(y) < 0.01]
    X_low_turn = [X[i] for i in range(len(X)) if abs(y[i]) < 0.01]
    n_low_turn = len(low_turn)
    n_turns = len(X_out)
    pick_size = min(n_low_turn,int(percent*n_turns))
    indices_picked = np.random.choice(n_low_turn,size=pick_size, replace=False)
    y_out.resize(n_turns + pick_size)
    for ind, val in enumerate(indices_picked):
        y_out[n_turns + ind] = low_turn[val]
        X_out.append(X_low_turn[val])
    return X_out, y_out

def main(_):
    X,y = GetData(csvFile)
    X,y = shuffle(X,y)
    train_X, valid_X, train_y, valid_y = train_test_split(X,y,train_size=0.95)
    train_X,train_y = equalizing_data(train_X,train_y, percent = 0)
    batch_size = FLAGS.batch_size
    valid_X = preprocessing_testing(valid_X)
    dir_save = saving_file(CurFolder)
    # Getting the NN model
    model = NVidia_Model()
    # Setting options for fit generator
    epoch_save_file = os.path.join(dir_save,'model.{epoch:04d}-{val_loss:.2f}.h5')
    weight_save_callback = ModelCheckpoint(epoch_save_file, monitor='val_loss', verbose=0,
                                           save_best_only=False, mode='auto',period=FLAGS.save_interval)
    stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='min')
    model.summary()
    model.fit_generator(generator_train(train_X,train_y,batch_size),
                        samples_per_epoch=FLAGS.sample_size,nb_epoch=FLAGS.epochs,
                        validation_data=(valid_X,valid_y),
                        callbacks=[weight_save_callback,stopping])
    # Saving model data
    model.save(os.path.join(dir_save,'model.h5'))
    with open(os.path.join(dir_save,'model.json'),'w') as json_file:
        json.dump(model.to_json(), json_file)

if __name__ == '__main__':
    tf.app.run()
