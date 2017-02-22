# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 07:39:37 2017

@author: admin
"""
import glob,os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

Curdir = os.getcwd()
imageFolder = os.path.join(Curdir,'Examples')

def image_compile(savename, imagePath, shape, Title, AngleFlag = True):
    file_list = glob.glob(imagePath)
    plt.figure(figsize=(10,10))
    for i in range(len(file_list)):
        filename = file_list[i]
        image = mpimg.imread(filename)
        plt.subplot(shape[0],shape[1], i+1)
        plt.imshow(image)
        plt.axis('off')
        index = filename.find('.png')
        if AngleFlag:
            try:
                angle = float(filename[index-5:index])
            except:
                angle = float(filename[index-4:index])
            plt.title('steering angle = '+str(angle))
    plt.suptitle(Title, fontsize=16)
    plt.savefig(os.path.join(imageFolder,savename))
    
outputCompilePath = os.path.join(imageFolder,'output*.png')
outputSaveName = 'output_compilation.png'
outputShape = (3,2)
outputTitle = 'Output augmented images set'
image_compile(outputSaveName,outputCompilePath,outputShape,outputTitle)

brightPath = os.path.join(imageFolder,'brightness*.png')
brightSaveName = 'Brightness_compilation.png'
brightShape = (5,2)
brightTitle = 'Brightness Augmentation'
image_compile(brightSaveName,brightPath,brightShape,brightTitle,AngleFlag=False)

shadowPath = os.path.join(imageFolder,'shadow*.png')
shadowSaveName = 'shadow_compilation.png'
shadowShape = (5,2)
shadowTitle = 'Shadow Augmentation'
image_compile(shadowSaveName,shadowPath,shadowShape,shadowTitle,AngleFlag=False)

transPath = os.path.join(imageFolder,'trans*.png')
transSaveName = 'translation_compilation.png'
transShape = (5,2)
transTitle = 'Translation Augmentation'
image_compile(transSaveName,transPath,transShape,transTitle)
#file_list = glob.glob()
#plt.figure(figsize=(8,8))
#for i in range(len(file_list)):
#    #with open(file_list[i]) as f:
#    filename = file_list[i]
#    index = filename.find('.png')
#    try:
#        anglestr = float(filename[index-5:index])
#    except:
#        anglestr = float(filename[index-4:index])
#    image = mpimg.imread(filename)
#    plt.subplot(3,2,i+1)
#    plt.imshow(image)
#    plt.axis('off')
#    plt.title('steering angle = '+str(anglestr))
#plt.suptitle('Output augmented images set', fontsize = 16)
#plt.savefig(os.path.join(imageFolder,'output_compilation.png'))